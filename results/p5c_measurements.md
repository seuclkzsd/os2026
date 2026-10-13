# P5c measurements

Final 16-thread tables of P5c: the integrated tree against main (regression, all data cases) and
against the tuned SLEPc baseline (head-to-head, the six P4 configurations). Sections 1-3 were
measured for this page on 2026-10-13/14; section 4 collects what the module fixers measured on their
own branches before the merge (not re-measured here); section 5 lists open issues.

## 1. Setup: what ran where

**Code.** "new" = `9201c2c` (head of `p5c-int-code`: `p5c-integration` 1bd75f9, i.e. main 2938cb4
plus the merges of asmkern, certint, anaint and blocktune, plus the code side of the P5c interface
requests: the blocktune block-size rule wired into the driver, InertiaCounter's deferred double
analysis, compare.py `hsolve-auto`, the rerun fixes and the 10 % margin for the rule's step to 8).
"main" = `2938cb4`. The branch `p5c-meas` adds only `tools/bench/ab_cases.py`,
`tools/bench/compare_merge.py` and these results on top of 9201c2c.

**Builds.** `tools/dev/build.sh release` (gcc 13.4, `-O3 -march=native`, OpenMP; cmake defaults of
each commit, so new has the BLAS hook with route `all` linked into hsolve and main has none), build
lane (cores 8-15, `HS_JOBS=4`): new in `~/os2026-wt/p5c-meas` on c005 (`HS_CLUSTER_NAME=p5c-meas`),
main in its own clone `~/os2026-wt/p5c-meas-main` on c003. The two CLI binaries were copied to
`~/p5c_int_p5c-meas/bin/hsolve_new` (md5 1ddf396c...) and `hsolve_main` (md5 dc5060d5...); every
node ran these copies in place (shared home, identical nodes: 2x Xeon Silver 4314). compare.py runs
`~/os2026-wt/p5c-meas/build/release/hsolve`, the same build as hsolve_new.

**Slot.** Every timed run held the node's measurement slot: cores 16-31 (NUMA node 1, the
16-core / 32 GiB judging configuration), `guard.py --exclusive`, one run at a time per node, 16
threads; each guard call <= 45 min. Evaluations (tools/eval) ran in the build lane. Nodes were
checked idle (`tools/dev/cluster_nodes.sh`) before each campaign; gpu02 was not used.

| campaign | node | slot time (2026-10-13) | content |
|---|---|---|---|
| A regression | c003 | 23:20-23:25 | 38 cases (all but the four below), 3 reps x 2 binaries |
| A regression | c005 | 23:20-23:31 | ss-ge87h76, ss-si34h36, syn-mem125-46 |
| A regression | gpu01 | 23:20-23:52, 23:52-00:04 | ss-3dspectralwave2 (r0; r1-r2; extra r3 after the failure in section 2.2) |
| B head-to-head | c003 | 23:25-23:28 | syn-mem27-46, yee-pec-box-1e5, wg-square-1e5 |
| B head-to-head | c005 | 23:31-23:33, 23:33-23:35 | ss-qa8fk-qa8fm, cav2d-rect-pert-1e5; yee-pec-box-1e5 [40, 200] |
| failure repro | c005 | 23:38-00:07 | ss-3dspectralwave2, hsolve_new with the event log, 6 runs (section 2.2) |
| failure repro | gpu01 | 00:05-00:35 | the same with hsolve_main, 5 runs |
| failure repro | c005 | 00:08-00:37 | the same with hsolve_main, 6 runs |

**Commands** (from `~/os2026-wt/p5c-meas`, `S=$HOME/p5c_int_p5c-meas`):

```bash
# A: interleaved ABBA (rep r: main,new for even r, new,main for odd r; reps outermost), default
#    configuration of both binaries (case config.json, --threads 16, CLI defaults: memory budget
#    14 GiB, certificate exact)
env/run.sh python3 tools/run/guard.py --exclusive --timeout 2700 --lock-wait 600 -- \
  python3 tools/bench/ab_cases.py run --bin main=$S/bin/hsolve_main --bin new=$S/bin/hsolve_new \
  --reps 0,1,2 --run-dir $S/A-c003 <cases>
env/run.sh python3 tools/run/guard.py --skip-busy-check --mem-gb 12 --timeout 3500 -- \
  python3 -m tools.eval.sweep $S/A-c003 --out $S/A-c003/eval.md       # every run, build lane
env/run.sh python3 tools/bench/ab_cases.py table --md $S/A-table.md \
  --json $S/p5c-regression.json $S/A-c003 $S/A-c005 $S/A-gpu01   # -> section 2, results/
# B: results/cmp-p16-best's method (tuned SLEPc profile "best", 5 interleaved reps, tools/eval on
#    every run, --mem-gb 20 -> hsolve --memory-budget-gb 18), one part per node, then merged
NUMPY_MADVISE_HUGEPAGE=0 env/run.sh python3 tools/run/guard.py --exclusive --timeout 2700 \
  --lock-wait 600 -- python3 tools/bench/compare.py --tag cmp-p5c-best-c003 --cores 16-31 \
  --mem-gb 20 --slepc-profile best --reps 5 --solvers hsolve,hsolve-auto,hsolve-nocert,slepc \
  --out-dir $S/B syn-mem27-46 yee-pec-box-1e5 wg-square-1e5
#   (c005: ss-qa8fk-qa8fm cav2d-rect-pert-1e5; and --tag cmp-p5c-best-interval
#    --target '{"mode":"interval","lower":40,"upper":200}' yee-pec-box-1e5)
python3 tools/bench/compare_merge.py --tag cmp-p5c-best --out-dir results \
  $S/B/cmp-p5c-best-c003.json@c003 $S/B/cmp-p5c-best-c005.json@c005
python3 tools/bench/compare_merge.py --tag cmp-p5c-best-interval --out-dir results \
  $S/B/cmp-p5c-best-interval.json@c005
```

## 2. Regression against main (campaign A)

All 42 data cases, 3 interleaved repetitions per binary (ss-3dspectralwave2: 4), medians of the
report's `time_total_s` (all of hs::solve after loading, the solver-time scope of compare.py) over
the valid runs. Valid = exit 0, not killed, success claimed and tools/eval/sweep.py passes (pooled
cross-run reference where the case has no comparable stored one). "numeric #" = the report's
factor_stats.factorizations (certificate counts on the main factorization object are in it, those
of a separate InertiaCounter - the large 3-D cases - are not); "block p" =
the reported block width (main: P4 rule; new: the blocktune rule for nearest targets).
Machine-readable: `results/p5c-regression.json`.

| case | n | type | node | main s [min-max] | new s [min-max] | new/main | valid main / new | numeric # main / new | block p main / new | certificate new |
|---|---|---|---|---|---|---|---|---|---|---|
| cav2d-rect-pert-1e5 | 99459 | GHEP | c003 | 0.344 [0.341-0.515] | 0.296 [0.292-0.301] | 0.861 | 3/3 / 3/3 | 4 / 3 | 1 / 2 | exact |
| cav2d-square-1e4 | 9976 | GHEP | c003 | 0.062 [0.062-0.064] | 0.048 [0.048-0.059] | 0.779 | 3/3 / 3/3 | 3 / 3 | 1 / 2 | exact |
| ss-2cubes_sphere | 101492 | HEP | c003 | 3.286 [3.241-3.353] | 2.874 [2.867-2.912] | 0.875 | 3/3 / 3/3 | 1 / 1 | 4 / 4 | exact |
| ss-3dspectralwave2 | 292008 | HEP | gpu01 | 356.447 [338.963-359.177] | 309.671 [304.166-320.267] | 0.869 (invalid runs: not compared) | 4/4 / 3/4 | 1 / 1 | 4 / 8 | exact |
| ss-ge87h76 | 112985 | HEP | c005 | 57.603 [57.247-60.104] | 54.728 [53.445-55.480] | 0.950 | 3/3 / 3/3 | 1 / 1 | 4 / 8 | exact |
| ss-helm3d01 | 32226 | HEP | c003 | 0.739 [0.729-0.773] | 0.548 [0.548-0.557] | 0.741 | 3/3 / 3/3 | 3 / 3 | 4 / 4 | exact |
| ss-qa8fk-qa8fm | 66127 | GHEP | c003 | 2.260 [2.251-2.302] | 1.689 [1.673-1.698] | 0.747 | 3/3 / 3/3 | 4 / 3 | 4 / 4 | exact |
| ss-si34h36 | 97569 | HEP | c005 | 40.767 [40.568-41.453] | 39.564 [38.083-39.793] | 0.970 | 3/3 / 3/3 | 1 / 1 | 4 / 8 | exact |
| ss-tem152078 | 152078 | GHEP | c003 | 5.109 [5.103-5.168] | 5.452 [5.440-5.472] | 1.067 **slower** | 3/3 / 3/3 | 4 / 4 | 4 / 4 | exact |
| ss-tem27623 | 27623 | GHEP | c003 | 0.744 [0.739-0.753] | 0.591 [0.588-0.599] | 0.794 | 3/3 / 3/3 | 4 / 4 | 4 / 4 | exact |
| syn-ghep-multiplicity-real-1e4 | 10000 | GHEP | c003 | 0.179 [0.178-0.192] | 0.134 [0.133-0.138] | 0.749 | 3/3 / 3/3 | 4 / 4 | 4 / 4 | exact |
| syn-ghep-near-singular-real-1e5 | 100000 | GHEP | c003 | 1.695 [1.689-1.709] | 1.286 [1.276-1.286] | 0.759 | 3/3 / 3/3 | 5 / 5 | 4 / 4 | exact |
| syn-ghep-zero-cluster-complex-1e4 | 10000 | GHEP | c003 | 0.387 [0.387-0.401] | 0.287 [0.287-0.292] | 0.741 | 3/3 / 3/3 | 4 / 3 | 4 / 4 | exact |
| syn-lap2d-100 | 10000 | HEP | c003 | 0.081 [0.081-0.093] | 0.066 [0.062-0.066] | 0.808 | 3/3 / 3/3 | 3 / 3 | 4 / 4 | exact |
| syn-lap3d-22 | 10648 | HEP | c003 | 0.235 [0.229-0.236] | 0.139 [0.139-0.141] | 0.593 | 3/3 / 3/3 | 3 / 3 | 4 / 4 | exact |
| syn-lap3d-aniso-1e5 | 97200 | HEP | c003 | 2.579 [2.539-2.690] | 2.084 [2.082-2.090] | 0.808 | 3/3 / 3/3 | 1 / 1 | 4 / 4 | exact |
| syn-mem125-46 | 97336 | HEP | c005 | 13.440 [13.321-13.740] | 11.036 [10.975-11.164] | 0.821 | 3/3 / 3/3 | 1 / 1 | 4 / 8 | exact |
| syn-mem27-46 | 97336 | HEP | c003 | 2.788 [2.710-2.807] | 2.374 [2.374-2.377] | 0.851 | 3/3 / 3/3 | 1 / 1 | 4 / 8 | exact |
| syn-multiplicity-complex-1e4 | 10000 | HEP | c003 | 0.263 [0.259-0.283] | 0.205 [0.202-0.214] | 0.777 | 3/3 / 3/3 | 3 / 3 | 4 / 4 | exact |
| syn-multiplicity-real-1e4 | 10000 | HEP | c003 | 0.124 [0.124-0.125] | 0.096 [0.095-0.103] | 0.772 | 3/3 / 3/3 | 3 / 3 | 4 / 4 | exact |
| syn-near-singular-complex-1e4 | 10000 | HEP | c003 | 0.321 [0.317-0.330] | 0.230 [0.229-0.238] | 0.718 | 3/3 / 3/3 | 4 / 4 | 4 / 4 | exact |
| syn-near-singular-complex-1e5 | 100000 | HEP | c003 | 3.463 [3.388-3.500] | 2.485 [2.472-2.513] | 0.718 | 3/3 / 3/3 | 4 / 4 | 4 / 4 | exact |
| syn-near-singular-real-1e4 | 10000 | HEP | c003 | 0.139 [0.132-0.147] | 0.103 [0.100-0.104] | 0.737 | 3/3 / 3/3 | 4 / 4 | 4 / 4 | exact |
| syn-wide-range-complex-1e4 | 10000 | HEP | c003 | 0.227 [0.225-0.231] | 0.182 [0.181-0.185] | 0.801 | 3/3 / 3/3 | 3 / 3 | 4 / 4 | exact |
| syn-wide-range-real-1e4 | 10000 | HEP | c003 | 0.107 [0.106-0.107] | 0.089 [0.087-0.092] | 0.836 | 3/3 / 3/3 | 3 / 3 | 4 / 4 | exact |
| syn-zero-cluster-complex-1e4 | 10000 | HEP | c003 | 0.470 [0.469-0.478] | 0.305 [0.304-0.309] | 0.649 | 3/3 / 3/3 | 5 / 5 | 4 / 4 | exact |
| syn-zero-cluster-real-1e4 | 10000 | HEP | c003 | 0.198 [0.193-0.199] | 0.130 [0.126-0.135] | 0.654 | 3/3 / 3/3 | 6 / 5 | 4 / 4 | exact |
| syn-zero-cluster-real-1e5 | 100000 | HEP | c003 | 1.753 [1.752-1.760] | 1.221 [1.220-1.234] | 0.697 | 3/3 / 3/3 | 6 / 6 | 4 / 4 | exact |
| wg-halfslab-1e4 | 9801 | GHIEP | c003 | 0.099 [0.098-0.110] | 0.076 [0.074-0.078] | 0.775 | 3/3 / 3/3 | 2 / 1 | 1 / 1 | none |
| wg-halfslab-1e5 | 99225 | GHIEP | c003 | 0.898 [0.857-0.899] | 0.750 [0.740-0.751] | 0.835 | 3/3 / 3/3 | 2 / 1 | 1 / 2 | none |
| wg-square-1e3 | 961 | GHIEP | c003 | 0.035 [0.033-0.040] | 0.025 [0.023-0.026] | 0.728 | 3/3 / 3/3 | 2 / 1 | 1 / 1 | none |
| wg-square-1e5 | 99225 | GHIEP | c003 | 0.597 [0.590-0.614] | 0.481 [0.478-0.487] | 0.805 | 3/3 / 3/3 | 2 / 1 | 1 / 2 | none |
| wg-square-1e6 | 998001 | GHIEP | c003 | 7.154 [7.089-7.214] | 5.941 [5.921-5.943] | 0.830 | 3/3 / 3/3 | 2 / 1 | 1 / 2 | none |
| wg-square-n289 | 289 | GHIEP | c003 | 0.024 [0.021-0.024] | 0.020 [0.016-0.023] | 0.849 | 3/3 / 3/3 | 2 / 1 | 1 / 1 | none |
| wg-square-pert-1e4 | 9801 | GHIEP | c003 | 0.091 [0.086-0.107] | 0.076 [0.072-0.085] | 0.840 | 3/3 / 3/3 | 2 / 1 | 1 / 1 | none |
| yee-bloch-1e4 | 10080 | HEP | c003 | 1.094 [1.090-1.141] | 0.831 [0.826-0.839] | 0.760 | 3/3 / 3/3 | 1 / 1 | 4 / 8 | exact |
| yee-pec-box-1e4 | 10064 | GHEP | c003 | 0.277 [0.270-0.292] | 0.200 [0.198-0.206] | 0.722 | 3/3 / 3/3 | 3 / 3 | 4 / 4 | exact |
| yee-pec-box-1e5 | 109744 | GHEP | c003 | 3.224 [3.187-3.291] | 2.922 [2.860-2.969] | 0.906 | 3/3 / 3/3 | 1 / 1 | 4 / 4 | exact |
| yee-pec-cube-16 | 10800 | HEP | c003 | 0.325 [0.307-0.340] | 0.213 [0.205-0.217] | 0.654 | 3/3 / 3/3 | 3 / 3 | 4 / 4 | exact |
| yee-pec-cube-32 | 92256 | HEP | c003 | 2.845 [2.789-2.888] | 2.255 [2.246-2.264] | 0.792 | 3/3 / 3/3 | 1 / 1 | 4 / 8 | exact |
| yee-photonic-rods-1e4 | 10125 | HEP | c003 | 1.113 [1.112-1.152] | 0.909 [0.904-0.921] | 0.817 | 3/3 / 3/3 | 1 / 1 | 4 / 8 | exact |
| yee-photonic-spheres-1e4 | 10125 | GHEP | c003 | 1.190 [1.184-1.199] | 0.973 [0.971-0.986] | 0.818 | 3/3 / 3/3 | 1 / 1 | 4 / 8 | exact |

Geometric mean of new/main over the 41 cases valid in every run: **0.786**; faster by > 3%: 39; slower by > 3%: 1 (ss-tem152078 1.067).
Not comparable (an invalid or missing run): ss-3dspectralwave2 (main: ok; new: r1: rc=2 verdict=not evaluated).

### 2.1 Notes

- **Geometric mean new/main 0.786** over the 41 cases valid in every run (0.787 over all 42 with
  ss-3dspectralwave2's valid runs, 0.869 there). By class: HEP 0.772 (24 cases), GHEP 0.808 (11),
  GHIEP 0.808 (7). Over the 18 cases where main needs more than 1 s: 0.832. The summed time of all
  42 cases drops from 514.8 to 453.6 s (0.881; dominated by ss-3dspectralwave2, ss-ge87h76 and
  ss-si34h36).
- Faster by more than 3 % on 39 of the 41 compared cases (plus ss-3dspectralwave2); ss-si34h36 at
  0.970 sits on the threshold.
- **Slower by more than 3 %: ss-tem152078 only, 1.067x** (5.109 -> 5.452 s, every run valid). The
  ordering model takes hsolve's ND (analysis 1.35 s) where main used SCOTCH (0.75 s); numeric
  factorization 2.27 -> 2.37 s, solves 1.17 -> 0.92 s (rep 0 reports). This is the deferred anaint
  finding (1.154x on the anaint branch); the faster solves of the merged tree halve the loss.
- Where the gains come from, as far as the reports show: one numeric factorization fewer where
  anaint's B test decides B without factorizing it (GHIEP 2 -> 1 on all seven wg-* cases, where B's
  Lanczos Ritz value shows it indefinite; cav2d-rect-pert-1e5, ss-qa8fk-qa8fm,
  syn-ghep-zero-cluster-complex-1e4 4 -> 3, where the B-HPD proof holds; also
  syn-zero-cluster-real-1e4 6 -> 5); the blocktune rule's width 8 on the
  3-D / high-fill nearest cases (syn-mem*, ss-ge87h76, ss-si34h36, ss-3dspectralwave2,
  yee-pec-cube-32, yee-bloch-1e4, yee-photonic-*) and width 2 on the low-k 2-D ones (cav2d-*,
  wg-square-1e5/1e6, wg-halfslab-1e5); the BLAS hook's faster solves everywhere (section 4.1).
  The small cases (n ~ 1e4) gain the most: 0.59-0.85x.
- Validity: 253 of the 254 runs valid (42 cases x 2 binaries x 3 reps, plus one extra repetition
  of ss-3dspectralwave2); every evaluation passed, against the stored reference or the pooled
  pairs of all runs of the case. The one invalid run is new-r1 of ss-3dspectralwave2 (2.2).

### 2.2 ss-3dspectralwave2: random failures at the default memory budget

new-r1 (gpu01) exited with status 2 after 205 s and wrote no result (its stderr was not kept;
ab_cases.py keeps it since 7abe7ac); the other three new runs and all four main runs passed. To
find the cause, repeated default runs with the event log followed (`--log`, otherwise the
campaign's command; in the slot, 16 threads; `~/p5c_int_p5c-meas/repro/`): hsolve_new 6x on c005,
hsolve_main 5x on gpu01 and 6x on c005. Under the CLI's default budget of 14 GiB (cap 13.49 GB
after the Krylov reservation) this case (n = 292008, complex, SCOTCH in-core estimate 26.9-28.3 GB)
runs with out-of-core factors, and each binary failed once more, in two different ways:

| run | binary | exit | time (s) | peak GiB | main analysis in-core / OOC estimate (MB) | counter in-core estimate (MB) | error |
|---|---|---|---|---|---|---|---|
| gpu01 A new-r1 | new | 2 | - | 12.25 | no log | no log | (stderr not kept; ~205 s, after the iteration) |
| c005 run1 | new | 0 | 303.0 | 12.56 | 27675 / 11044 | 11007 | |
| c005 run2 | new | 0 | 305.7 | 12.81 | 27545 / 11287 | 11223 | |
| c005 run3 | new | 0 | 290.5 | 11.71 | 26869 / 10580 | 10574 | |
| c005 run4 | new | 0 | 309.6 | 15.11 | 27919 / 11477 | **13628** | |
| c005 run5 | new | 2 | - | 11.50 | 27365 / 11120 | **14504** | MUMPS factorization ran out of memory: INFO(1)=-19, INFO(2)=55094667 (working memory above the ICNTL(23) cap) |
| c005 run6 | new | 0 | 307.8 | 13.01 | 28096 / 11326 | 11278 | |
| c005-main run1 | main | 2 | - | 3.67 | 27123 / **14167** | - | predicted factorization memory 27123 MB (out-of-core 14167 MB) exceeds the budget of 13489.5 MB |
| c005-main run2 | main | 0 | 349.8 | 13.67 | 27151 / **13422** | 11182 | |
| c005-main run3 | main | 0 | 346.9 | 12.30 | 27659 / 11173 | 11138 | |
| c005-main run4 | main | 0 | 341.8 | 12.81 | 27087 / 10876 | 10815 | |
| c005-main run5 | main | 0 | 351.7 | 12.68 | 27714 / 11282 | 11137 | |
| c005-main run6 | main | 0 | 358.7 | 13.23 | 28257 / 11730 | 11503 | |
| gpu01-main run1 | main | 0 | 353.3 | 12.33 | 27672 / 11075 | 11008 | |
| gpu01-main run2 | main | 0 | 359.4 | 13.03 | 27342 / 11138 | 11085 | |
| gpu01-main run3 | main | 0 | 356.7 | 12.39 | 27420 / 11166 | 11133 | |
| gpu01-main run4 | main | 0 | 361.3 | 12.52 | 27950 / 11204 | 11187 | |
| gpu01-main run5 | main | 0 | 366.3 | 11.94 | 28322 / 11355 | 11301 | |

1. **Certificate counter over the cap** (new: c005 run5; gpu01 new-r1 failed at the same point).
   The 10 pairs had converged (lanczos nconv = 10 at 194 s). The certificate's InertiaCounter
   re-analyses the main factorization's pivot order in core; its MUMPS estimate was 10.6-11.3 GB in
   four of the six logged new runs but 13.6 and 14.5 GB in the other two. At 14.5 GB the counter's
   factorization fails (INFO(1) = -19), the exception ends the solve with exit 2 and the converged
   pairs are lost; at 13.6 GB (run4) it ran, with a peak RSS of 15.1 GiB, above the 14 GiB budget.
2. **Main factorization over the budget** (main: c005-main run1). The main analysis' out-of-core
   estimate was 14.2 GB (13.4 GB in another main run, otherwise 10.9-11.7 GB) and the run stopped
   after 19 s.

Both are MUMPS estimates that scatter by up to ~30 % from run to run (SCOTCH's random order, and
the counter's re-analysis of it) against a budget with little headroom. Outliers appeared in both
binaries (main: 2 of 11 logged main analyses; new: 2 of 6 counter analyses), and
`CountSource::count`, which builds and runs the counter, is the same code on main (steps.cpp). This
points to a pre-existing fragility rather than a P5c regression, but the samples (new: 2 failures in
10 runs, main: 1 in 15) are too small to exclude that P5c made mode 1 more likely. Either way,
neither binary is robust on this case at the default budget (section 5). Times of the valid runs:
new 290-320 s, main 339-366 s.

## 3. Head-to-head against tuned SLEPc (campaign B)

Full tables: `results/cmp-p5c-best.md` / `.json` (five nearest configurations) and
`results/cmp-p5c-best-interval.md` / `.json` (yee-pec-box-1e5, interval [40, 200]); same method,
profiles and repetitions as `results/cmp-p16-best*` (P4, gpu01, code 19a6e1a). Every run of every
solver was valid (5/5). Speedup = SLEPc median solver time / hsolve median solver time (> 1: hsolve
faster).

| case | target | SLEPc s (P4 / P5c) | hsolve s (P4 -> P5c) | hsolve vs SLEPc (P4 -> P5c) | hsolve-auto vs SLEPc (P5c; auto's certificate) | hsolve-nocert vs SLEPc (P4 -> P5c) |
|---|---|---|---|---|---|---|
| syn-mem27-46 | k=12 @ 0 | 3.49 / 3.48 | 2.76 -> 2.39 | 1.26x -> **1.45x** | **1.96x** (randomized) | 1.89x -> 2.34x |
| yee-pec-box-1e5 | k=10 @ 52.85 | 3.35 / 3.36 | 3.24 -> 2.95 | 1.03x -> **1.14x** | 1.14x (exact) | 1.45x -> 1.60x |
| wg-square-1e5 | k=18 @ -0.527 | 0.89 / 0.89 | 0.59 -> 0.49 | 1.50x -> **1.84x** | 1.82x (none: GHIEP) | 1.51x -> 1.83x |
| ss-qa8fk-qa8fm | k=10 @ 12.82 | 1.48 / 1.50 | 2.32 -> 1.68 | 0.64x -> **0.89x** | 0.89x (exact) | 0.81x -> 1.18x |
| cav2d-rect-pert-1e5 | k=6 @ 0.1418 | 0.22 / 0.21 | 0.34 -> 0.30 | 0.64x -> **0.71x** | 0.71x (exact) | 0.79x -> 0.94x |
| yee-pec-box-1e5 | [40, 200] | 10.46 / 10.30 | 4.42 -> 3.81 | 2.37x -> **2.70x** | 2.72x (slicing counts) | 2.41x -> 2.71x |

- SLEPc reproduces its P4 (gpu01) times within 2 % on c003/c005, so the P4 and P5c columns compare.
- Where the time went (P5c hsolve rows of cmp-p5c-best.md): syn-mem27-46 runs the rule's width 8
  (iterate 0.55 -> 0.27 s, solves 0.47 -> 0.24 s for 48 instead of 44 right-hand sides);
  yee-pec-box-1e5 keeps width 4 (the 10 % margin) and gains in the solves (0.91 -> 0.72 s, BLAS
  hook); ss-qa8fk-qa8fm and cav2d-rect-pert-1e5 drop from 4 to 3 numeric factorizations (the B-HPD
  proof replaces B's factorization, anaint) and solve faster (hook); wg-square-1e5 drops from 2 to 1
  factorization (B shown indefinite without factorizing it) and runs at width 2.
- hsolve-auto: randomized check on syn-mem27-46 (certificate 0.24 s vs 0.90 s exact); exact counts
  everywhere else (yee-pec-box-1e5: the 10x margin keeps exact; qa8 / cav2d: no cheaper rigorous
  route; the GHIEP case has no certificate; slicing counts anyway).
- Remaining gap: ss-qa8fk-qa8fm and cav2d-rect-pert-1e5 with the certificate on. Without it hsolve
  is 1.18x / 0.94x of SLEPc; the certificate's counts are 2 of the 3 factorizations
  (0.40 s / 0.06 s certificate time against SLEPc's single factorization).

## 4. Module measurements (from the fixers' reports, measured on the module branches)

Not re-measured for this page; the numbers come from the module reports
(`p5c_interface_requests.md` of the P5c run) and were taken on each module's own branch, in the
16-thread measurement slot unless stated, before the four modules were merged.

### 4.1 asmkern (fix-asmkern-p5c @ e67a141; gpu01 slot, 16 threads, ~10 min slot time)

- Machine reference: dgemm 78 GF/s on 1 core, 747 GF/s on 16 (two AVX-512 FMA units per core);
  read bandwidth 36 GB/s at 4 cores, no growth beyond.
- A one-column solve stops scaling after 4 threads: it streams the factors, and 47-70 % of it runs
  above MUMPS' L0 layer on one core. With the hook's route `all` it is 21-31 % faster per right-hand
  side.
- compare.py --solvers hsolve, hook (route all) against plain hsolve, 5 interleaved reps, every run
  valid: syn-mem27-46 1.046x, yee-pec-box-1e5 1.131x, wg-square-1e5 1.145x, ss-qa8fk-qa8fm 1.134x,
  cav2d-rect-pert-1e5 1.115x, yee-pec-box-1e5 [40, 200] 1.164x. All of it in the solves (MUMPS solve
  time -18..-30 %), numeric factorization unchanged.
- Drop-in BLAS libraries (implementer's screening, 3 reps): rebuilt OpenBLAS 0.97-1.07x, LIBXSMM
  0.97-1.02x, BLIS 0.46-0.57x on the 2-D cases; none adopted.

### 4.2 certint (fix-certint-p5c @ 57f98ca; gpu02 slot, 16 threads)

- certificate=auto against exact on all 32 HEP/GHEP cases, 5 reps: 0.99-1.00x where both use exact
  counts; faster where auto picks the randomized check (syn-mem27-46 1.29x, syn-mem125-46 1.19x,
  yee-pec-cube-32 1.09x) or single-precision counts (yee-pec-cube-16 1.05x).
- Against tuned SLEPc (compare.py, 5 reps, hsolve-auto through a scratch case root): syn-mem27-46
  1.63x, yee-pec-box-1e5 1.02x (1.15x with single counts before the 10x margin), ss-qa8fk-qa8fm
  0.65x, cav2d-rect-pert-1e5 0.63x.
- ss-ge87h76: auto predicted 27.28 s randomized vs 27.23 s exact and chose exact; measured 25.1 s vs
  29.8 s (one case; count_ratio and preference left unchanged).

### 4.3 anaint (fix-anaint-p5c @ 0c424c6; gpu02 slot, 16 threads)

- Final binary against main on all 42 data cases, 3 interleaved reps: geometric mean 0.891, faster
  on 32 cases, slower only on ss-tem152078 (5.070 -> 5.851 s, 1.154x; ND analysis 1.32 s vs
  SCOTCH 0.71 s). All runs valid.
- syn-zero-cluster-real-1e5: 1.45x slower -> 0.83x of main after the null-pivot cluster rule.
- Ordering cost model (OrderingModel / OrderingPricer, bench/anaint_fix_fit.cpp replaying the
  driver's choice): 4.3 % mean excess over the best ordering in-sample, 5.0 % leave-one-case-out; a
  SCOTCH-preference rule that fixed ss-tem152078 raised it to 5.3 % and was reverted.
- hsolve ND vs SCOTCH factorization flops: yee-pec-box-1e5 8.1e10 vs 6.9e10, ss-ge87h76 6.1e12 vs
  5.1e12, ss-tem152078 7.6e10 vs 7.1e10.

### 4.4 blocktune (fix-blocktune-p5c @ 8f29115; gpu01 slot, 16 threads)

- Rule fix (p >= 4 for nev >= 8): the old p = 2 pick on low-fill multiplicity cases was slower,
  0.182 vs 0.162 s and 0.288 vs 0.236 s. Over 61 case/nev points the chosen-to-best width time ratio
  went from 1.11 to 1.035 (geometric mean), never slower than p = 1.
- End to end (bench/blocktune_fix_e2e.cpp, 12 cases, 5 interleaved runs, 155 runs all valid; rule
  applied through --block-size, not yet wired): 1.03-2.45x faster than p = 1; 0.98-1.19x against the
  P4 width (syn-mem27-46 1.05x, yee-pec-cube-32 1.12x, yee-bloch-1e4 1.06x, cav2d-square-1e4 1.19x;
  ss-2cubes_sphere at p = 8 1.6 % slower).
- Block-method layout variants (ncv extra / keep): geometric mean -0.9 %, defaults kept.

### 4.5 Integration (p5c-int-code; gpu01 slot, 16 threads; from its commit messages)

- Wired rule against the P4 width, one binary with the hook (b1cfbef): 1.02-1.08x where the rule
  moves 4 -> 8 on multiple eigenvalues or 1 -> 2 at k = 6, 7; 0.96-0.98x where it moves 4 -> 8 on
  distinct eigenvalues (ss-2cubes_sphere, yee-pec-box-1e5, yee-bloch-1e4); geometric mean 1.01x.
- 10 % margin for the step to 8 (9201c2c; --block-size 0 vs 4, 5 interleaved runs, 20 definite
  nearest cases): ss-2cubes_sphere, yee-pec-box-1e5, syn-lap3d-aniso-1e5 back at width 4 (1.020x,
  1.058x, 1.008x faster than at 8); geometric mean against the P4 width 1.001 -> 1.017.
- Deflated reruns (cf9f077, certificate_fault_drop = 1): rerun phase at width 4 instead of 8
  1.05-1.33x faster; yee-pec-box-1e5 with one pair removed reruns in 4.65-4.81 s (before: 1000
  restarts, ~200 s).

## 5. Open issues

1. **ss-3dspectralwave2 is not robust at the default 14 GiB budget, in both binaries** (2.2): new
   failed in 2 of 10 runs through the certificate counter (INFO -19 after the pairs had converged),
   main in 1 of 15 through the main factorization's out-of-core estimate. Most likely pre-existing
   (scattering MUMPS estimates against a tight budget), but a judged run of this size could fail.
   Candidate fixes for the owners (driver / certint; anaint / factor): a counter factorization that
   fails for memory should not discard converged pairs (fall back to another certificate route,
   retry out of core or with a different order, or report the run as uncertified); the SCOTCH seed
   (anaint interface request) would make the orderings, and with them the estimates, reproducible;
   whether the CLI's default budget (14 GiB) should follow the 32 GB judging machine is a lead
   decision. The same runs show the budget is not a hard limit: peak RSS 15.1 GiB with a counter
   estimated 13.6 GB against the 13.5 GB cap.
2. **ss-tem152078 is 1.067x slower than main** (the only case beyond 3 %): the ordering model's ND
   analysis (1.35 s) against main's SCOTCH (0.75 s); deferred anaint finding (needs a better
   predictor of ND work, or the reproducible SCOTCH).
3. **The default (exact certificate) still trails tuned SLEPc on ss-qa8fk-qa8fm (0.89x) and
   cav2d-rect-pert-1e5 (0.71x)**; without the certificate hsolve is 1.18x / 0.94x. The
   certificate's two count factorizations are the gap; certint's interface request (factorize
   B - mu diag(B) for a proven kappa, making the randomized check rigorous on general B) targets it.
4. **certificate=auto** (hsolve-auto) changes the head-to-head only on syn-mem27-46 (1.96x vs 1.45x
   for exact); it keeps exact counts on the other nearest cases. The default stays exact; switching
   it is the user's decision (certint report).
5. ss-3dspectralwave2 and the other 3-D high-fill cases run the rule's width 8, which was
   calibrated in core; on this out-of-core case it is still 0.869x of main, so no action, but the
   out-of-core regime was not part of the rule's calibration.
6. Not covered by this page: other thread counts than 16, CPUs without AVX-512 (asmkern), the
   certint auto choice on ss-ge87h76 (module report only), SLEPc on the regression cases.
