# P7 "membudget": a memory budget from the machine, and single precision that cannot return a wrong answer

**Questions.** (1) The fixed default `memory_budget_gb = 14` sent a 1e6-unknown 3-D case out of core
on a 32 GB machine (P6: syn-mem27-100 274.6 s instead of 125.1 s in core). Can the budget follow the
memory the run may actually use, without ever letting a run exceed it? (2) `factor_precision =
single` returned invalid results or errors on 4 of 11 P6 cases. Can it be made safe, and is it worth
choosing automatically when the double factors do not fit in core?

**Answers.**

* **Budget.** `memory_budget_gb` now defaults to `"auto"`: max(0.5, min(0.85 L, L - 1)) GiB of the
  smallest memory limit L of the process (memory cgroup v1/v2, `HSOLVE_MEMORY_LIMIT_GB` - exported
  by `tools/run/guard.py` with the run's RSS cap -, RLIMIT_AS / RLIMIT_DATA, MemTotal). A configured
  number still wins. In the 32 GiB slot the two 3-D 1e6 cases now factorize in core:
  **syn-mem27-100 1.83x (199.1 -> 109.0 s), yee-pec-box-1e6 1.71x (157.6 -> 92.0 s) faster than the
  14 GB default** (medians of 3, all valid); under a 16 GiB cap (the 16 GB judging machine) they go
  out of core as before, at the default's speed, and stay valid; the 2-D wg-square-1e6 and the five
  1e5 head-to-head cases make the same decisions and take the same time.
* **Never above the limit.** Over 103 slot runs under caps of 6 to 32 GiB nothing was killed; the
  peak RSS stayed at or below 0.83 of the cap (0.99 of the budget before the re-plan below, 0.89
  after it). A cap too small even for out-of-core
  factors (syn-mem27-100 under 8 GiB) is refused after the analysis (OutOfMemoryBudget, exit 2,
  peak 2.6 GiB), never started. Three mechanisms keep the plan honest: the cap is re-planned after the
  analysis by what the analysis kept resident (0.5-1 GB at n = 5e5-1e6, unseen before); an in-core
  factorization that still fails for memory is retried out of core; MUMPS gets the cap as ICNTL(23).
* **Single precision is safe.** Every refined application is checked, a single factor that is
  singular even without null-pivot detection ends the attempt, and a result that is not accepted
  too; the solve then starts again with double factors on the same pivot order. All four P6
  failures (syn-mem27-46, syn-ghep-near-singular-real-1e5, syn-near-singular-complex-1e5,
  syn-zero-cluster-real-1e5) now return valid results (double, after a 0.6-1.5 s single attempt),
  the seven others stay single with eigenvalues within 2e-15 of double's.
* **precision auto** (single only when double does not fit in core within the budget but single
  does) is 1.10x on syn-mem27-100 at a 16 GiB cap but 0.84-1.04x on yee-pec-box-1e6, where the
  refinement next to the null space often fails and the attempt costs 24 s. It stays **opt-in**;
  `factor_precision = double` remains the default (section 6).

## 1. What changed (code)

| piece | where | what |
|---|---|---|
| memory limits | `include/hsolve/util.hpp`, `src/util/memory_limits.cpp` | `detect_memory_limits`: cgroup v2 `memory.max` / v1 `memory.limit_in_bytes` and `hierarchical_memory_limit` along the path of `/proc/self/cgroup` (a container's own cgroup at the mount root), `HSOLVE_MEMORY_LIMIT_GB`, RLIMIT_AS / RLIMIT_DATA when finite, MemTotal; the smallest wins. MemAvailable is not used: the decision must not depend on what other processes hold at the moment. |
| budget | `include/hsolve/solver.hpp`, `src/driver/memory_budget.cpp`, `src/driver/solver.cpp` | `memory_budget_gb` default `"auto"` (= 0; config, CLI `--memory-budget-gb GiB\|auto`, Python); `resolve_memory_budget`; report `memory{budget_gb, source, limit_gb, limit_source, limits[]}`; a configured budget above the limit warns. |
| guard | `tools/run/guard.py` | exports `HSOLVE_MEMORY_LIMIT_GB` = min(--mem-gb, an enclosing guard's value): 32 in the slot, 12 in the lane, 6 for `tools/dev/test.sh`, 8 for `tools/bench/compare.py`'s solver guards (default `--mem-gb 8`: pass `--mem-gb 32` there for 1e6 cases). |
| re-plan | `src/driver/solver.cpp`, `SparseFactorization::set_max_memory_mb` | the factorization cap (planned before the analysis) shrinks by what the analysis kept resident, before the in-core / out-of-core decision (event `memory_replan`). |
| out-of-core retry | `FactorOptions::ooc_on_memory_error` | with `out_of_core = auto`, an in-core factorization that fails for memory (INFO -13/-19, or a workspace the retries cannot grow under the cap) is retried out of core (`factor_stats.ooc_switches`). |
| checked inexact factors | `DriverShiftInvert` (`src/driver/internal.hpp`, `steps.cpp`), `ShiftInvertOperator::last_apply_converged` | every refined application of a single / BLR factor must converge (or stagnate within tol * inner_tol_factor), else `InexactFactorFailure`; a single factor flagged singular is refactorized without detection and only an exactly singular one ends the attempt. |
| fallback | `src/driver/solver.cpp` | on `InexactFactorFailure` or an unaccepted result the solve restarts after the diagnosis with double factors on the abandoned attempt's pivot order (`set_pivot_order`); report `precision{requested, used, blr, decision, fallback, fallback_reason, attempt_s}`. |
| precision auto | `choose_precision` (`steps.cpp`) | after the (double) analysis: if its in-core estimate exceeds the cap, a single object is analysed on the same pivot order and taken if its in-core estimate fits. |
| memory timeline | `src/util/log.cpp` | every JSON-lines event carries `rss_mb` and `peak_rss_mb`. |

Tests: `tests/cpp/test_membudget.cpp` (fake cgroup v1/v2 trees, environment, rlimits, the budget
formula, config round trip, the out-of-core retry, the re-plan, a valid out-of-core run below the
in-core estimate), `tests/cpp/test_membudget_single.cpp` (the operator check; single at a
well-separated shift stays single; the P6 failures' small analogues - a shift 1e-9 ||K|| from a
delocalized eigenvalue, a near-singular complex prescribed spectrum, a near-singular GHEP - fall back
and are valid; GHIEP; precision auto on the estimates and end to end), `tests/py/test_guard.py`
(the exported cap, nested guards), `tests/py/test_membudget_binding.py`. Whole suite: 930 ctest +
351 pytest green on gpu02 (release, warning-free).

## 2. Method

* **Where.** gpu02, measurement slot: cores 16-31, 16 threads, `guard.py --exclusive`, one run at a
  time, 2026-10-14. Every run's RSS cap is `--mem-gb` (32 = the judging machine with 32 GB, 16 =
  with 16 GB, 6-24.5 for the calibration), which the guard enforces (kill above it) and exports as
  `HSOLVE_MEMORY_LIMIT_GB`. gpu02 has a cgroup-v1 memory controller without a limit and 251 GiB
  MemTotal, so the automatic budget of every P7 run below came from the guard's declaration
  (`report.memory.source = "env"`). Correctness-only runs (single precision on the P6 cases, memory
  timelines) used the build/test lane (cores 8-15, 8 threads, no timing).
* **Binaries.** base = 9201c2c (fixed 14 GB budget; it ignores the variable). P7: the 1e6 and 1e5
  measurements ran 54d0ed8 (budget, guard, checks, fallback, auto, timeline); the calibration of
  section 3 d73f8d1 (the same without the timeline); the confirmation runs of sections 3, 4 and 6 the
  final code bf4dea8, which adds the cap re-plan after the analysis and the two refinements of the
  inexact-factor check (section 6). Each binary is a copy of `build/release/hsolve` of that commit
  (`tools/dev/build.sh release`, the BLAS hook linked).
* **Cases.** syn-mem27-100 (HEP, n = 1e6, 27-point 3-D stencil, sigma = 0, k = 12), yee-pec-box-1e6
  (GHEP, n = 1027733, empty PEC cavity, sigma = 52.9 above the gradient null space, k = 10;
  generated for P7 with `tools.gen.yee.make_homogeneous_case("yee-pec-box-1e6",
  YeeGrid((92, 71, 54), (1.0, 0.75, 0.55)), "generalized", root)`, exact reference, MUMPS estimate
  16.5 GB in core / 8.7 GB out of core at 8 threads, so it fits the 28 GiB rule), wg-square-1e6
  (GHIEP, n = 998001), syn-mem27-80 (HEP, n = 512000, `make_memory_case("syn-mem27-80",
  (80, 80, 80), 1, 0.02, root)`) for the small caps, and the five 1e5 head-to-head cases.
* **Repetitions.** 3 interleaved per (case, variant), the variant order rotated per repetition;
  1 for the calibration and the confirmations. Every run was evaluated with `tools/eval/sweep.py`
  (backward errors from the matrices, completeness against the exact references).
* **Commands** (from the repository root on gpu02). `meas.sh RUN REPS CASE... -- VARIANT...` (a
  throw-away loop, not committed) runs, for repetition r = 1..REPS and every case, each variant
  `NAME:BINARY:CAP_GB[:EXTRA]` once - the order rotated by r - with the per-run command below, into
  the raw layout of `tools/eval/sweep.py`:

```
# per run (meas.sh RUN REPS CASE... -- NAME:BINARY:CAP_GB[:EXTRA]...):
python3 tools/run/guard.py --exclusive --timeout 1800 --lock-wait 3600 --mem-gb CAP \
    --summary RUN/CASE/guard-NAME-rR.json -- BINARY --A CASE/A.npz [--B CASE/B.npz] \
    --config CASE/config.json --out RUN/CASE/NAME-rR -q --log RUN/CASE/NAME-rR.jsonl [EXTRA]
# 1e5 head-to-head, then the 1e6 cases:
meas.sh h2h 3 syn-mem27-46 yee-pec-box-1e5 wg-square-1e5 ss-qa8fk-qa8fm cav2d-rect-pert-1e5 -- \
    base:bin-base/hsolve:32 auto32:bin-p7/hsolve:32
meas.sh big 3 syn-mem27-100 yee-pec-box-1e6 wg-square-1e6 -- base:bin-base/hsolve:32 \
    auto32:bin-p7/hsolve:32 auto16:bin-p7/hsolve:16 "auto16p:bin-p7/hsolve:16:--precision auto"
# calibration: syn-mem27-80 under 6 / 8 / 10.5 GiB, syn-mem27-100 under 8 / 12 / 24.5 GiB
meas.sh cal 1 syn-mem27-80 -- c6:bin-p7/hsolve:6 c8:bin-p7/hsolve:8 c105:bin-p7/hsolve:10.5
meas.sh cal 1 syn-mem27-100 -- c8:bin-p7/hsolve:8 c12:bin-p7/hsolve:12 c245:bin-p7/hsolve:24.5
# final code (bf4dea8): calibration again, the P6 cases, one confirmation run of the 1e6 cases
meas.sh cal2 1 syn-mem27-80 -- c6:bin-final/hsolve:6 c8:bin-final/hsolve:8 c105:bin-final/hsolve:10.5
meas.sh cal2 1 syn-mem27-100 -- c8:bin-final/hsolve:8 c12:bin-final/hsolve:12 c245:bin-final/hsolve:24.5
meas.sh p6slot 1 syn-mem27-46 syn-ghep-near-singular-real-1e5 syn-near-singular-complex-1e5 \
    syn-zero-cluster-real-1e5 yee-pec-box-1e5 wg-square-1e5 ss-qa8fk-qa8fm cav2d-rect-pert-1e5 -- \
    "single:bin-final/hsolve:32:--precision single" double:bin-final/hsolve:32
meas.sh final 1 syn-mem27-100 yee-pec-box-1e6 wg-square-1e6 -- auto32:bin-final/hsolve:32 \
    auto16:bin-final/hsolve:16 "auto16p:bin-final/hsolve:16:--precision auto"
# evaluation (build/test lane):
python3 tools/run/guard.py --skip-busy-check --mem-gb 12 --timeout 3500 -- \
    python -m tools.eval.sweep RUN --root DATA --out eval_RUN.md
```

Memory timelines (lane, no timing): `hsolve ... --log run.jsonl` under `guard.py --skip-busy-check
--mem-gb 6`; every event carries `rss_mb` / `peak_rss_mb`.

## 3. Calibration: peak RSS against the budget

The plan (DESIGN §3.6-3) keeps three predicted quantities within the budget: the resident memory
when it plans, the Krylov reservation, and MUMPS' working memory under the ICNTL(23) cap (0.95 of
what is left). The rest of the limit L must hold what the plan does not predict. To see how much
that is, the largest cases ran under caps that put the plan at its edge: syn-mem27-80 (in core 7.2-7.4
GB, out of core 2.6-3.3 GB) under 6, 8 and 10.5 GiB, and syn-mem27-100 (in core 18.2-19.4 GB, out of
core 7.0-8.7 GB) under 8, 12 and 24.5 GiB - 10.5 and 24.5 were the smallest caps whose budget let the
factors in core before the re-plan below. Each ran twice: with d73f8d1 (cap planned before the
analysis) and with the final bf4dea8.

| case | cap L (GiB) | auto budget (GiB) | factorization cap MB: planned -> after the analysis | MUMPS estimate in core / out of core, MB | factors | MUMPS used MB | peak RSS GiB | peak / budget | peak / L | total s | valid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| syn-mem27-80 | 6 | 5.00 | 4306 (d73f8d1) / 4296 -> 3517 (bf4dea8) | 7340 / 3274; 6899 / 2645 | out of core / out of core | 2528 / 2264 | 4.96 / 4.47 | 0.99 / 0.89 | 0.83 / 0.75 | 88.4 / 83.6 | yes / yes |
| syn-mem27-80 | 8 | 6.80 | 6140 / 6140 -> 5359 | 6903 / 2607; 7357 / 2830 | out of core / out of core | 2242 / 2413 | 4.53 / 4.48 | 0.67 / 0.66 | 0.57 / 0.56 | 84.8 / 89.6 | yes / yes |
| syn-mem27-80 | 10.5 | 8.93 | 8309 / 8307 -> 7547 | 7213 / 2878; 7402 / 2952 | in core / in core | 6132 / 6269 | 7.14 / 7.25 | 0.80 / 0.81 | 0.68 / 0.69 | 38.2 / 39.1 | yes / yes |
| syn-mem27-100 | 8 | 6.80 | 5573 / 5573 -> 4487 | 18775 / 8694; 18954 / 7173 | refused / refused (exit 2) | - | 2.63 / 2.57 | - | 0.33 / 0.32 | - | - |
| syn-mem27-100 | 12 | 10.20 | 9041 / 9041 -> 7907 | 18228 / 7045; 18846 / 7296 | out of core / out of core | 5965 / 6195 | 8.46 / 8.44 | 0.83 / 0.83 | 0.70 / 0.70 | 202.8 / 201.9 | yes / yes |
| syn-mem27-100 | 24.5 | 20.83 | 19880 / 19881 -> 18747 | 19447 / 7724; 19018 / 7403 | in core / **out of core** | 16379 / 6266 | 16.21 / 8.71 | 0.78 / 0.42 | 0.66 / 0.36 | 119.3 / 206.6 | yes / yes |

Pairs of values: d73f8d1 (before the re-plan) / bf4dea8 (final). Estimates differ between the two
runs because SCOTCH's orderings vary from run to run (factor.hpp).

* **Where the unpredicted memory is** (event timeline, syn-mem27-80 under 6 GiB, lane): resident
  210 MB when the plan is made, 751 MB after the analysis (MUMPS keeps its symbolic data and a copy
  of the input: 0.54 GB that the plan did not see), +2974 MB during the out-of-core factorization -
  exactly MUMPS' out-of-core estimate on top of what was there -, 1.5 GB during the Krylov run (the
  factors are on disk), 4.3 GB peak during the certificate's inertia-only counts (the counter's
  in-core working memory, 2.95 GB, beside the Krylov basis). At 16 threads the same run peaked at
  4.96 GiB: L0-OpenMP keeps per-thread work areas. Hence the **re-plan** in the final code: after
  the analysis the cap shrinks by what the process grew (`memory_replan`: 0.80-0.82 GB on syn-mem27-80, 1.14-1.19 GB on syn-mem27-100).
* **The re-plan's price.** It makes the in-core decision honest, and more conservative where MUMPS
  over-estimates: syn-mem27-100 under 24.5 GiB went in core before it (peak 16.2 GiB, 119 s) and
  goes out of core with it (207 s). In core it now needs a cap of about 25 GiB - far below the
  32 GiB machine (section 4: cap 25.3 GB after the re-plan against an estimate of 18-19 GB).
* **The fraction.** Peak RSS reached 0.66-0.99 of the budget before the re-plan and 0.42-0.89 after
  it, and at most 0.83 of the cap L. The
  budget is a good predictor of the peak, so the rest of L only needs to hold the unpredicted part:
  0.15 L (at least 1 GiB) holds it with room for a harness process (a judge that loads the matrices
  in Python before calling hsolve) - the auto budget is 27.2 GiB of 32, 13.6 of 16, 5.0 of 6.
  Nothing is gained by more: the measured cases make the same in-core / out-of-core decisions with
  0.9 L, and a larger fraction only shrinks the margin.
* **In core vs out of core at the edge.** MUMPS' in-core estimate (INFOG(16)) exceeds what the
  factorization uses (INFOG(21)) by 15-18% (19447 vs 16379 MB, 7213 vs 6132 MB), its out-of-core
  estimate by 13-30%; the decision keeps the estimate, the cap catches the rest (INFO -19 -> the
  out-of-core retry, never seen in these runs).
* **Too small for out of core.** syn-mem27-100 under 8 GiB: budget 6.8 GiB, cap 5.6 GB (4.5 GB after
  the re-plan), out-of-core estimate 7.2-8.7 GB - refused after the analysis (exit 2,
  OutOfMemoryBudget, 8 s, peak 2.6 GiB), never started. It needs a cap of about 11 GiB to run out of
  core (valid under 12 GiB, peak 8.4 GiB). A 512000-unknown 3-D case (syn-mem27-80, factors 6.1-6.3 GB)
  runs out of core under 6 GiB and stays valid.

## 4. The 1e6 cases: 14 GB default vs auto at 32 GiB vs auto at 16 GiB

| case | variant | valid | factors | budget GiB | total s, median [min-max] | factor | iterate | certificate | solves s (RHS) | peak GiB (max) | max eta |
|---|---|---|---|---|---|---|---|---|---|---|---|
| syn-mem27-100 | base (14 GB) | 3/3 | out of core | 14 | **199.07** [191.15-199.78] | 38.8 | 104.0 | 55.3 | 102.3 (168) | 8.68 | 8.6e-13 |
| syn-mem27-100 | auto, 32 GiB | 3/3 | in core | 27.2 | **109.01** [102.50-110.85] | 35.2 | 17.7 | 55.3 | 16.1 (168) | 15.28 | 1.1e-11 |
| syn-mem27-100 | auto, 16 GiB | 3/3 | out of core | 13.6 | **198.66** [190.81-212.09] | 39.3 | 102.6 | 56.1 | 100.9 (168) | 8.44 | 7.8e-13 |
| syn-mem27-100 | auto + precision auto, 16 GiB | 3/3 | in core, single | 13.6 | 180.68 [179.32-183.75] | 22.5 | 98.4 | 59.6 | 88.5 (1336) | 9.67 | 3.7e-14 |
| yee-pec-box-1e6 | base (14 GB) | 3/3 | out of core | 14 | **157.56** [154.91-159.10] | 32.2 | 78.3 | 46.4 | 75.9 (136) | 7.39 | 8.9e-13 |
| yee-pec-box-1e6 | auto, 32 GiB | 3/3 | in core | 27.2 | **91.95** [89.35-92.70] | 29.8 | 14.4 | 47.3 | 12.1 (136) | 13.00 | 2.2e-12 |
| yee-pec-box-1e6 | auto, 16 GiB | 3/3 | out of core | 13.6 | **157.26** [156.58-163.92] | 32.3 | 78.7 | 46.1 | 76.4 (136) | 7.46 | 1.2e-12 |
| yee-pec-box-1e6 | auto + precision auto, 16 GiB | 3/3 | 1x in core single, 2x fallback to double out of core | 13.6 | 178.86 [151.28-181.23] | 45.8 | 86.6 | 46.9 | 78.2 (136) | 8.20 | 1.2e-12 |
| wg-square-1e6 | base (14 GB) | 3/3 | in core | 14 | 5.94 [5.93-5.94] | 1.2 | 3.7 | 0.0 | 2.4 (64) | 1.93 | 2.3e-13 |
| wg-square-1e6 | auto, 32 GiB | 3/3 | in core | 27.2 | 5.91 [5.86-5.96] | 1.2 | 3.6 | 0.0 | 2.3 (64) | 1.94 | 1.4e-12 |
| wg-square-1e6 | auto, 16 GiB | 3/3 | in core | 13.6 | 5.93 [5.92-5.97] | 1.2 | 3.6 | 0.0 | 2.4 (64) | 1.93 | 2.3e-13 |
| wg-square-1e6 | auto + precision auto, 16 GiB | 3/3 | in core (double fits) | 13.6 | 5.94 [5.85-5.95] | 1.2 | 3.7 | 0.0 | 2.4 (64) | 1.91 | 1.5e-12 |

Phases are medians in seconds (factor = analyses + numeric factorizations: analysis 4.4-6.0 s,
numeric factorization 24-29 s in core and 27-33 s out of core); peak RSS is the guard's process-tree
peak. P7 binary 54d0ed8; the final code's confirmation runs are in the bullets.

* **32 GiB (auto32).** The budget is 27.2 GiB; both 3-D cases now factorize in core (cap 26.4 GB
  against estimates of 18.1 / 15.5 GB): **syn-mem27-100 1.83x (199.1 -> 109.0 s), yee-pec-box-1e6
  1.71x (157.6 -> 92.0 s)**, medians of 3, all valid. The final code re-plans the cap to 25.2 / 25.8
  GB (the analysis kept 1.2 / 0.9 GB resident) and decides the same: 102.8 / 91.1 s, valid. The gain is the solves: 102 -> 16 s and 76 -> 12 s
  (out-of-core solves read the factors back from disk every time); the factorization is 10-12% faster
  in core. Peak RSS 15.3 / 13.0 GiB, well inside 32. The certificate (two inertia-only counts,
  46-56 s) is now half of the run - the next lever for these cases, outside P7.
* **16 GiB (auto16, the judging machine with 16 GB).** Budget 13.6 GiB, both 3-D cases out of core,
  as with the old 14 GB default and at its speed (198.7 vs 199.1 s, 157.3 vs 157.6 s); peak 8.4 /
  7.5 GiB. Valid in every run; the final code: 204.3 / 154.7 s, valid.
* **wg-square-1e6** (2-D, GHIEP, 1.3 GB in core) is in core under every budget: 5.9 s for all four.
* **precision auto at 16 GiB (auto16p)** - see section 6.

## 5. The 1e5 head-to-head cases are unchanged

| case | base (14 GB): total s, median [min-max] | auto, 32 GiB | ratio | factors (both) | solves (RHS) | peak GiB base / auto | valid |
|---|---|---|---|---|---|---|---|
| syn-mem27-46 | 2.46 [2.41-2.54] | 2.46 [2.43-2.49] | 1.00 | in core | 48 / 48 | 0.86 / 0.87 | 6/6 |
| yee-pec-box-1e5 | 2.93 [2.90-3.05] | 3.00 [2.93-3.12] | 0.98 | in core | 84 / 84 | 0.92 / 0.95 | 6/6 |
| wg-square-1e5 | 0.51 [0.50-0.51] | 0.50 [0.49-0.61] | 1.02 | in core | 62 / 62 | 0.24 / 0.24 | 6/6 |
| ss-qa8fk-qa8fm | 1.73 [1.72-1.76] | 1.71 [1.69-1.78] | 1.01 | in core | 100 / 100 | 0.61 / 0.63 | 6/6 |
| cav2d-rect-pert-1e5 | 0.31 [0.31-0.33] | 0.31 [0.31-0.32] | 1.00 | in core | 36 / 36 | 0.25 / 0.22 | 6/6 |

Same decisions (in core everywhere, the same orderings and solve counts), time ratios 0.98-1.02
(noise), all 30 runs valid. The automatic budget is 27.2 GiB here instead of 14: only the cap given
to MUMPS changes, and these factors need under 1 GB.

## 6. Single precision: safe, and when auto pays

**The checks** (final code, DESIGN §3.4b / §3.6-3):

1. Every operator application of a single (or BLR) factor is refined in double (at most 10
   corrections, target 0.1 * tol * inner_tol_factor = 1e-12 relative) and checked
   (`DriverShiftInvert`): an application whose refinement stopped without converging is accepted
   only if its relative residual is within the Krylov tolerance tol * inner_tol_factor (1e-11):
   stagnation at the attainable accuracy just misses the target (wg-square-1e5: 1.08e-12), while
   eps_single * cond(K) >= 1 leaves it at 1e-6 and above. Otherwise `InexactFactorFailure`.
2. A single factor flagged singular (threshold 1e-4 ||K||) is refactorized at the shift without
   null-pivot detection; check 1 then decides. Only a factor that is singular even so (the complex
   embedding splitting a pair) ends the attempt. (The first version ended every flagged attempt;
   cav2d-rect-pert-1e5, 16 null pivots, then fell back although its refinement converges.)
3. The result must be accepted (every pair within tol, the certificate - double counts - passed).

Any failure abandons the attempt (its time stays in the phases, `report.precision.attempt_s`) and
the solve restarts after the diagnosis with double factors on the same pivot order.

**The P6 cases with `--precision single`** (final code bf4dea8; slot, 16 threads, 1 run each, all 16
runs valid by tools/eval; and the 11 P6 cases in the lane, all valid):

| case | P6 (V1, base 1bd75f9) | P7 single: outcome | attempt s | total s single / double | RHS single / double |
|---|---|---|---|---|---|
| syn-mem27-46 | invalid: eta up to 2.8e-5, certificate failed (0/3) | fallback: refinement ends at 0.51 (lane runs: 2.4e-5, 8.2e-6) | 1.37 | 3.13 / 2.40 | 48 / 48 |
| syn-ghep-near-singular-real-1e5 | invalid: 1 pair (0/3) | fallback: flagged, refactorized without detection, refinement ends at 6.2e-3 | 1.28 | 2.08 / 1.29 | 108 / 108 |
| syn-near-singular-complex-1e5 | error rc = 2 "exactly singular" | fallback: singular without detection | 1.27 | 3.12 / 2.49 | 101 / 100 |
| syn-zero-cluster-real-1e5 | timeout | fallback: flagged, refactorized without detection, refinement diverges (14.9) | 0.61 | 1.43 / 1.24 | 120 / 120 |
| yee-pec-box-1e5 | valid, 0.59x | single | - | 5.26 / 2.97 | 439 / 84 |
| wg-square-1e5 | valid, 0.35x | single | - | 1.34 / 0.49 | 294 / 62 |
| ss-qa8fk-qa8fm | valid, 0.59x | single | - | 2.90 / 1.74 | 409 / 100 |
| cav2d-rect-pert-1e5 | valid, 0.40x | single (flagged, refactorized without detection) | - | 0.65 / 0.33 | 137 / 36 |
| syn-mem125-46, ss-tem152078, yee-bloch-1e4 (lane) | valid | single | - | - | - |

Where single stays single, its eigenvalues agree with double's to |dlambda| / (|lambda| + scale)
<= 2.0e-15 (ss-tem152078; <= 4e-17 elsewhere), unlike P6's unrefined V2 (2.3e-8 on ss-tem152078).
Single is still 1.7-2.7x slower than double where it works (refinement: 4-5x the right-hand sides),
and a fallback costs its attempt (0.6-1.4 s here). **Its only purpose is memory.**

**precision auto at 16 GiB** (section 4 table; auto16p):

* syn-mem27-100: the double in-core estimate (18.8-19.1 GB) exceeds the cap (12.5 GB), single
  (9.6 GB) fits: single factors in core in 3/3 runs, **180.7 s against 198.7 s out of core (1.10x)**,
  valid, peak 9.7 GiB (8.4 out of core). The factorization halves (14.9 vs 32.5 s numeric); the
  Krylov phase does not gain: 1336 refined right-hand sides (8 per application) cost 88.5 s against
  100.9 s for 168 out-of-core ones.
* yee-pec-box-1e6: single in core in 1 of 3 runs (151.3 s, 1.04x); in the other two an application
  ended at 1.5e-6 and 9.0e-12 relative residual, the attempt was abandoned after 23.6-23.8 s, and the
  double out-of-core run made the total 178.9-181.2 s (0.87x). Next to the curl-curl null space
  (sigma = 52.9) the refinement contracts slowly and stagnates around 1e-11: the final code accepts
  stagnation within 1e-11, which would have kept the 9.0e-12 run single, but its confirmation run
  ended at 1.14e-11 and fell back as well.
* Final code (bf4dea8), one confirmation run each, valid: syn-mem27-100 single in core, 186.4 s
  against 204.3 s out of core (1.10x); yee-pec-box-1e6 fallback after 23.9 s, 183.1 s against
  154.7 s (0.84x).
* wg-square-1e6: double fits in core, auto keeps double.

**Trade-off.** Single in core instead of double out of core gains at most ~10% on these 3-D cases -
the factorization halves, but the refined solves cost about what the out-of-core solves do - and
loses ~13% when the refinement fails, which happens at shifts near a large null space. Auto is
therefore **not made the default**: `factor_precision = double` stays the default, `auto` is
available where memory is the constraint (it also turns into a run what out of core would refuse:
when double does not fit even out of core while single fits in core).

## 7. Conclusion and recommendations

* **The default budget follows the machine** (`memory_budget_gb = "auto"`, 0.85 of the smallest
  limit, at most the limit less 1 GiB). On a 32 GB machine the 1e6-unknown 3-D cases factorize in
  core and run **1.71-1.83x faster** than with the fixed 14 GB (final code: 102.8 / 91.1 s against
  199.1 / 157.6 s); on a 16 GB machine they run out of core exactly as before; everything else
  (wg-square-1e6, the five 1e5 head-to-head cases) makes the same decisions in the same time. A
  configured number still wins; old configs keep working; the report says which budget and why.
* **No run exceeds its limit.** 103 slot runs under caps of 6-32 GiB: none killed, peak RSS at most
  0.83 of the cap (0.89 of the budget with the final re-plan); a cap too small even for out-of-core
  factors is refused cleanly after the analysis. The guard now declares its cap
  (`HSOLVE_MEMORY_LIMIT_GB`), so guarded runs plan with the slot's 32 GiB, the lane's 12, the test
  suite's 6 - and `tools/bench/compare.py`'s default `--mem-gb 8` for its solver runs: **pass
  `--mem-gb 32` to compare.py for 1e6 cases** (tools owner: consider making the slot cap its default).
* **Single precision never returns a wrong answer any more**: the four P6 failures are valid (via
  the double fallback, at 1.15-1.6x the double time), the seven others stay single with double's
  eigenvalues. It remains slower than double (1.7-2.7x where it works); its purpose is memory.
* **precision auto stays opt-in** (default double): single in core instead of double out of core is
  1.10x on syn-mem27-100 and 0.84-1.04x on yee-pec-box-1e6 (refinement stagnating at the 1e-11
  acceptance band next to the null space). Revisit only with more 3-D cases at 16 GiB, or a cheaper
  refinement.
* **Next lever for the 1e6 cases in core:** the certificate's two inertia-only counts take 46-56 s,
  about half of the in-core run (`certificate = auto` / the randomized check were not part of P7).
* **For the judging machine:** with a cgroup (container) or a VM, the budget comes from the cgroup or
  MemTotal; a harness that keeps large data in another process can declare a smaller limit through
  `HSOLVE_MEMORY_LIMIT_GB`.
