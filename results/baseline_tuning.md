# SLEPc baseline tuning (P4, module baseline)

Goal: make the comparison target as strong as possible, so that a speedup claim of hsolve is a claim
against the fastest **valid** open-source configuration at equal core count: SLEPc 3.26 Krylov-Schur +
shift-and-invert + MUMPS 5.8.2 (env `os2026-ref`, ParMETIS / PT-SCOTCH available).

Measured 2026-10-11 on 2 cores (40,41) of the shared machine (load average 40-44 on 96 hardware
threads). This is noisy: the same configuration's median moved by up to 30 % between runs an hour
apart (ss-qa8fk-qa8fm default: 4.80 s, later 3.52 s; hsolve on wg-square-1e5: 1.93 s, later 2.54 s).
That is why every decision rests on interleaved repetitions and medians, never on single runs.

Revised after review (branch fix-baseline-p4). The first version stored a wg-square-1e5 profile that
fails tools/eval in about one run of five: 2 of 8 runs in the review, 2 of 6 in the re-tuning. Its
5 passing confirmation runs could not show that. A configuration that fails one run in five passes
5 of 5 with probability 0.33. The tuner now confirms randomly failing cases with
20 runs per configuration, and wg-square-1e5 was re-tuned that way. compare.py now runs tools/eval
on every result. The head-to-head below was re-measured with it, 5 interleaved repetitions.

## Bottom line

* Tuned SLEPc vs SLEPc's default (2 cores, interleaved medians): **1.13x** syn-mem27-46, **1.20x**
  ss-qa8fk-qa8fm, **1.07x-1.29x** wg-square-1e5 (two sessions), **1.94x** cav2d-rect-pert-1e5,
  **1.03x** (noise) for the interval case.
* SLEPc's default is **not valid** on yee-pec-box-1e5 (in all 5 confirmation runs it misses a copy
  of a multiple eigenvalue and leaves a pair above tol). On wg-square-1e5 it fails at random (1 of
  6 runs in one session, 0 of 21 in another). The cheapest valid configuration on yee-pec-box-1e5
  costs as much as the invalid default (0.98x).
* **On wg-square-1e5, SLEPc fails at random in almost every configuration**: 1 and 2 ranks, with
  and without BLR. The stored profile is the fastest configuration that passed every one of at
  least 20 runs: `1x2` + BLR + QAMD, 63 of 63 runs valid.
* **hsolve vs the tuned baseline at 2 cores** (same scope, every run of both solvers valid; table
  below): with the certificate on, hsolve runs at 0.51x / 0.67x / 0.46x / 0.66x of SLEPc's speed on
  syn-mem27-46 / yee-pec-box-1e5 / ss-qa8fk-qa8fm / cav2d-rect-pert-1e5, so it is *slower* there.
  It is at parity on wg-square-1e5 (1.04x) and 1.49x faster on the interval case. With the
  certificate off it reaches 0.78x / 0.80x / 0.62x / 0.83x / 1.07x (wg-square-1e5) / 1.42x
  (interval). The gap comes from hsolve's analysis and its extra certificate factorizations (see
  "Lessons for hsolve").

## Method (tools/bench/tune_slepc.py)

* Configuration = ranks x threads layout + binding + PETSc options (`tools/bench/slepc_profile.py`;
  MUMPS controls carry the spectral transform's prefix: `-st_mat_mumps_icntl_7`). Option values
  may be k-rules (`2k`, `k+4`), so a profile serves any k. Options a layout cannot use (parallel
  analysis and ICNTL(13) on one rank, slicing partitions not dividing the rank count) are dropped
  with the layout.
* Greedy coordinate descent from the default (1 rank x P threads, SLEPc/MUMPS defaults), group
  by group. Each candidate is measured against the incumbent and the incumbent moves to the
  fastest valid candidate when it is more than 3 % faster. Within the 3 % the simpler
  configuration wins. The groups:
  accuracy (only while the incumbent is invalid: `-eps_tol tol/10`, `-eps_nev k+2/k+4`, `-eps_ncv 3k`)
  -> layout (1xP ... Px1, bound/unbound) -> MUMPS ordering ICNTL(7) AMD/AMF/SCOTCH/PORD/METIS/QAMD and
  parallel analysis ICNTL(28)=2 with PT-SCOTCH/ParMETIS -> layout again -> ICNTL(48)=0, ICNTL(58)=1,
  ICNTL(13)=1, ICNTL(8)=0 -> BLR ICNTL(35)=2 with CNTL(7) 1e-12/1e-10 -> `-eps_ncv` k+8/2k/3k/4k ->
  `-eps_krylovschur_restart` 0.3/0.7 -> `-eps_purify 0` -> GHIEP: `-eps_gen_non_hermitian` -> prune
  (drop each option again). Interval targets: `-eps_krylovschur_partitions`,
  `-eps_krylovschur_nev` 10/20/60 per shift, orderings, ICNTL(48)/(58).
* **Valid** = the result passes `tools/eval` at the case tolerance (backward error with
  conservative 2-norm estimates, completeness against the case reference, duplicates). The
  convergence test stays EPS_CONV_NORM at the case tol. Only the accuracy group may tighten it.
  Options PETSc never used make a trial invalid (a misspelt option would silently tune nothing).
* Metric: the solver time from `-log_view`: EPSSetUp (MUMPS analysis + factorization) + EPSSolve
  + the eigenvector computation EPSSolve defers (purification). It has the same scope as hsolve's
  `time_total_s`. Python/MPI start-up, case loading and output are excluded
  (`tools/bench/slepc_log.py`).
* Confirmation: the default, the descent's start (with `--from-profile`: the stored profile), the
  best and the runners-up, 5 interleaved repetitions (rotating order). Medians decide. A
  configuration counts as valid only if **every** one of its runs passed, screening included.
* Randomly failing cases: when some configuration of the case was valid in some runs and invalid
  in others, the case is "flaky". This is detected in the tuning, or read from a stored profile of
  the case at any core count. Every configuration still valid after the 5 repetitions then gets
  interleaved runs up to 20, and it must pass all of them. A configuration that fails in a quarter
  of its runs passes 5 of 5 with probability 0.24, and 20 of 20 with 0.003. The flag is stored in
  the profile (`tuned.flaky`). Only wg-square-1e5 is flaky so far.
* The 2-core run: `tune-p2` (all groups except BLR/prune for the nearest cases), then
  `tune-p2-ext` (`--from-profile --groups blr,purify,prune`), then `--rescore` with the final
  selection rule. After the review: `--rescore` again (adds the flaky flags), and wg-square-1e5
  re-tuned with the 20-run rule: `tune-p2-fix` (full descent from the stored profile) and
  `tune-p2-fix2` (prune). Raw trials: `results/raw/tune-p2*/<case>/` (not tracked). Profiles:
  `tools/bench/slepc_profiles.json`.

## Best configuration per case (2 cores)

Solver time in s, median of 5 interleaved runs (wg-square-1e5: 21). Phases (analysis, numeric,
solves, ortho) are the best configuration's medians.

| case | type | target | P | default (s) | best (s) | speedup | best configuration | analysis | numeric (#) | solves (#/s) | ortho | valid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| syn-mem27-46 | HEP | k=12 | 2 | 5.285 | 4.675 | 1.13x | `1x2 eps_ncv=2k st_mat_mumps_icntl_7=4` | 0.498 | 1.822 (1) | 36 / 2.290 | 0.050 | yes |
| yee-pec-box-1e5 | GHEP | k=10 | 2 | 7.194 (invalid) | 7.337 | 0.98x | `1x2 eps_nev=k+4 eps_purify=0` | 1.241 | 1.857 (1) | 50 / 4.097 | 0.168 | yes |
| wg-square-1e5 | GHIEP | k=18 | 2 | 2.700 (invalid 1/6) | 2.513 | 1.07x | `1x2 st_mat_mumps_cntl_7=1e-12 st_mat_mumps_icntl_35=2 st_mat_mumps_icntl_7=6` | 0.076 | 0.175 (1) | 75 / 1.669 | 0.389 | yes (21/21) |
| ss-qa8fk-qa8fm | GHEP | k=10 | 2 | 3.523 | 2.940 | 1.20x | `1x2 eps_ncv=3k eps_purify=0 st_mat_mumps_icntl_7=4` | 0.556 | 0.661 (1) | 44 / 1.448 | 0.166 | yes |
| cav2d-rect-pert-1e5 | GHEP | k=6 | 2 | 1.104 | 0.570 | 1.94x | `2x1/core eps_purify=0 st_mat_mumps_icntl_7=2` | 0.051 | 0.077 (1) | 22 / 0.377 | 0.037 | yes |
| yee-pec-box-1e5 | GHEP | [40, 200] (37 eigenvalues) | 2 | 23.800 | 23.047 | 1.03x | `2x1 eps_krylovschur_nev=60` | 1.286 | 4.038 (2) | 194 / 15.591 | 1.818 | yes |

(`1x2` = 1 MPI rank x 2 OpenMP threads; `/core` = ranks pinned to disjoint cores of the guard's
cpuset and OpenMP threads to cores. ICNTL(7): 2 AMF, 4 PORD, 6 QAMD; default 7 = automatic, which
picked METIS here. BLR: ICNTL(35)=2 with compression tolerance CNTL(7).)

Confirmation runs (median [min-max] s; interval case from `tune-p2`, wg-square-1e5 from
`tune-p2-fix`/`tune-p2-fix2`, the others from `tune-p2-ext`):

- **syn-mem27-46**: default 5.285 [5.127-6.012]; best 4.675 [4.493-5.219]; + BLR 1e-10 4.904; + BLR 1e-12 5.059
- **yee-pec-box-1e5**: default 7.194 [7.076-7.625] INVALID in 5/5 (1 pair above tol, 1 eigenvalue copy missing);
  `eps_nev=k+4 eps_purify=0` 7.337 [6.840-8.626]; + BLR 1e-12 8.115; `eps_nev=k+4` (purification on) 9.179
- **wg-square-1e5** (valid runs / runs of each configuration in its session, screening included):
  - The previously stored profile `2x1/core` BLR 1e-12 ICNTL(13)=1 QAMD was valid in only 4 of 6
    runs of `tune-p2-fix`: 5 eigenvalues missing in one run, 8 of 18 pairs above tol in another.
    The review measured 6 of 8.
  - `tune-p2-fix`: every 2-rank finalist failed at random. `2x1/core` BLR AMD (the descent's
    winner) was valid in 20 of 21 runs, median 2.227 [1.725-3.000], and lost 6 eigenvalues once.
    Its GNHEP variant was valid in 3 of 6 (duplicate vectors).
  - `tune-p2-fix` 1-rank results: `1x2` BLR QAMD (with the no-op ICNTL(13)=1) 2.382 [1.958-3.769],
    21 of 21. `1x2/core` of it 2.514, 21 of 21. Default `1x2` 3.065 [2.628-3.774], 21 of 21.
  - `tune-p2-fix2` (prune, 20 runs each): `1x2` BLR QAMD 2.513 [1.939-3.253], 21 of 21 (the
    profile). With the no-op ICNTL(13)=1 it ran 2.478, 21 of 21. Without BLR it ran 2.323
    [1.804-3.619] but was valid in only 19 of 21 (2 eigenvalues missing twice). Without the
    ordering (METIS) 3.042, valid in 18 of 21. Default `1x2` 2.700, valid in 5 of 6.
  - The `1x2` BLR QAMD computation was valid in 63 of 63 runs across the two sessions. With
    BLR, SLEPc is 8 % slower here than without it, but without it SLEPc missed eigenvalues. That
    may be chance: no configuration on this case is known to be failure-free.
- **ss-qa8fk-qa8fm**: default 3.523 [3.351-3.704]; best 2.940 [2.867-3.293]; + ICNTL(58)=1 2.935 (same, more options)
- **cav2d-rect-pert-1e5**: default 1.104 [1.081-1.273]; best 0.570 [0.556-0.643]; + BLR 1e-10 0.559 (within noise)
- **yee-pec-box-1e5 [40, 200]**: default 23.800 [21.917-26.184]; best 23.047 [20.496-29.535]; `1x2 nev=60`
  26.565; `2x1/core nev=60` 27.441. All within the noise.

## What the knobs do (screening, 2 cores)

| knob | effect | evidence (screening trials, solver s) |
|---|---|---|
| MUMPS ordering, 2-D cases | METIS (the automatic choice) costs 0.5-0.7 s of analysis. QAMD/AMF cost 0.06-0.1 s with equal numeric time: **the largest single gain** | wg-square-1e5 analysis 0.72 -> 0.09 s (QAMD); cav2d analysis 0.46 -> 0.06 s (AMF) |
| MUMPS ordering, 3-D cases | PORD halves the analysis at similar factor cost. AMD/AMF/QAMD are 2-5x worse in numeric and solve | syn-mem27-46 analysis 1.08 -> 0.49 s, total 4.94 -> 4.70; yee-pec-box-1e5 PORD is slower (numeric 2.6 vs 1.6 s) |
| `-eps_purify 0` (generalized) | saves about k solves (purification applies OP to every vector). Valid on all GHEP cases, invalid on GHIEP (wg-square: pairs above tol) | yee-pec-box 68 -> 50 solves; cav2d 29 -> 22; ss-qa8fk 54 -> 44 |
| `-eps_ncv` | 2k-3k beat the default max(2k, k+15) on some cases, 4k always loses | syn-mem27-46 2k: 41 -> 36 solves; ss-qa8fk 3k: 58 -> 54 |
| layout at 2 cores | 2x1 (MPI) wins on the 2-D cases, where solves and vector operations dominate; 1x2 wins on 3-D. On wg-square-1e5, 2x1 is faster but fails at random | cav2d 1.50 -> 1.19 s; syn-mem27-46 2x1 5.27 vs 1x2 4.94 |
| parallel analysis ICNTL(28)=2 | slower than the best sequential ordering at 2 ranks (PT-SCOTCH 0.19-0.42 s vs QAMD/AMF 0.06-0.09 s) | worth re-testing at 16 ranks, where the analysis is serial and 1.18 s |
| BLR ICNTL(35)=2 | no speed gain at n = 1e5 (higher numeric, slightly fewer flops in the solves). CNTL(7)=1e-10 breaks accuracy on yee-pec-box. On wg-square-1e5 at 1x2, 1e-12 is 8 % slower than no BLR, yet it is the only variant that passed all of its 63 runs | syn-mem27-46 4.68 -> 4.90/5.06; wg-square-1e5 2.32 (19/21 valid) -> 2.51 (21/21) |
| ICNTL(48)=0, ICNTL(8)=0, ICNTL(58)=1, restart 0.3/0.7, GNHEP for GHIEP | no gain, or invalid (restart 0.3/0.7 and GNHEP on wg-square: missing eigenvalues / duplicate vectors) | negative results kept out of the profiles |
| `-eps_krylovschur_nev` (slicing) | 10/20 per shift are **invalid** (6 of 37 pairs above tol). 60 is valid, no faster than the default 40 | interval case |

## Head-to-head at 2 cores (tools/bench/compare.py --slepc-profile best)

Same scope: hsolve `time_total_s` against SLEPc's `-log_view` setup + solve + vectors. Medians of 5
interleaved runs (interval case: 3), `cmp-p2-best-fix` / `cmp-p2-best-fix-interval`. SLEPc ran each
case's own tuned profile.

Accuracy basis: compare.py ran tools/eval on every result at the case tol. A run counts only if it
exited normally, claimed success and passed. **All 75 + 9 runs passed** (hsolve, hsolve-nocert
and SLEPc). The
completeness check is certified only where the case has a complete reference:
syn-mem27-46 and yee-pec-box-1e5 (nearest and interval; discrete-exact references). For
wg-square-1e5, ss-qa8fk-qa8fm and cav2d-rect-pert-1e5 the reference is an incomplete ARPACK
solve (18 / 20 / 6 pairs). There, tools/eval checks consistency with it (status "consistent"):
the result must contain the reference's k nearest eigenvalues (all of them where the reference
has no more than k), but an eigenvalue ARPACK missed would go unnoticed. On these three cases the hsolve and
SLEPc eigenvalue sets agree to <= 2e-14 in |lambda-mu|/(|mu|+||A||/||B||).

| case | hsolve (s) | hsolve-nocert (s) | SLEPc tuned (s) | hsolve speedup | nocert speedup | hsolve analysis / numeric (#) / solves | SLEPc analysis / numeric (#) / solves |
|---|---|---|---|---|---|---|---|
| syn-mem27-46 | 9.57 | 6.26 | 4.88 | 0.51x | 0.78x | 1.44 / 5.32 (3) / 36 in 2.58 s | 0.52 / 1.92 (1) / 36 in 2.37 s |
| yee-pec-box-1e5 | 9.98 | 8.30 | 6.65 | 0.67x | 0.80x | 1.43 / 5.21 (3) / 50 in 3.29 s | 1.23 / 1.88 (1) / 50 in 3.46 s |
| wg-square-1e5 | 2.54 | 2.47 | 2.63 | 1.04x | 1.07x | 0.07 / 0.36 (2) / 57 in 1.38 s | 0.07 / 0.17 (1) / 76 in 1.76 s |
| ss-qa8fk-qa8fm | 7.64 | 5.70 | 3.53 | 0.46x | 0.62x | 1.03 / 3.47 (4) / 50 in 2.50 s | 0.63 / 0.82 (1) / 44 in 1.81 s |
| cav2d-rect-pert-1e5 | 1.00 | 0.80 | 0.66 | 0.66x | 0.83x | 0.04 / 0.36 (4) / 22 in 0.39 s | 0.07 / 0.08 (1) / 22 in 0.43 s |
| yee-pec-box-1e5 [40, 200] | 18.01 | 18.91 | 26.89 | 1.49x | 1.42x | 1.35 / 6.16 (3) / 113 in 8.58 s | 1.30 / 4.04 (2) / 239 in 18.67 s |

The first version of this table had hsolve 1.20x faster on wg-square-1e5. That was
against the randomly failing 2-rank BLR profile, and in a less loaded hour (hsolve 1.93 s then,
2.54 s now). The SLEPc configuration without BLR that sometimes misses eigenvalues is about 8 %
faster than the stored profile (2.32 vs 2.51 s in one session). Against it, hsolve would be at
roughly 0.96x.

Wall time including start-up and loading (guard) is a different measure. SLEPc's Python + MPI
start-up and per-rank case loading add 2.4-2.7 s. With the certificate on, hsolve's wall time is
**slower** than SLEPc's on syn-mem27-46 (0.77x), yee-pec-box-1e5 (0.89x) and ss-qa8fk-qa8fm (0.77x).
It is faster on wg-square-1e5 (1.93x), cav2d-rect-pert-1e5 (2.67x) and the interval case (1.63x).
Only hsolve-nocert is faster in wall time on every case (1.16x / 1.07x / 1.99x / 1.03x / 3.21x /
1.55x). Claims must use the solver-time column.

Lessons for hsolve (input for the analysis / certfast modules, not changed here):
1. PORD analysis on 3-D cases: SLEPc's 0.52 s vs hsolve's 1.44 s analysis on syn-mem27-46, with
   the same single-factorization cost (1.75-1.92 s).
2. The certificate's extra factorizations are most of the gap. hsolve's numeric time is 2.8-4.5x
   SLEPc's because it runs 3-4 factorizations. hsolve-nocert runs 1-2 and reaches 0.62-0.83x.
3. hsolve's solve phase is close to parity. Per solve it is within 10 % of SLEPc on four cases,
   and 22 % slower on ss-qa8fk-qa8fm (50 ms vs 41 ms, with 50 vs 44 solves). On the interval case
   hsolve needs half the solves.

## Pitfalls found in the previous baseline (they affect the older cmp4/cmp4b numbers)

1. **Option values were dropped**: the old `baseline_slepc.py` passed only the tokens starting with
   `-` to PETSc, so `-mat_mumps_icntl_7 5` lost its value. Without the `st_` prefix it would have
   been ignored anyway. Options are now passed through in order, and unused ones are reported.
2. **Solver time missed the purification**: EPSSolve defers the eigenvector computation to the
   first EPSGetEigenpair. On generalized problems that is about k extra solves outside the old
   `time_solve_s` (yee-pec-box-1e4: 12 of 43 solves). `time_solve_s` now includes it, which made
   the old comparisons slightly favourable to SLEPc.
3. **Spectrum slicing with partitions silently used PETSc's native Cholesky**: the per-partition
   solvers are configured from the options database, so the code-set MUMPS choice did not reach
   them. Setup took 190 s instead of 7.5 s (yee-pec-box-1e5 [40, 200]). The solver choice now also
   goes into the database.
4. **`mpiexec --bind-to core` binds outside the guard's cpuset**: a run pinned to CPUs 40,41 got
   its ranks bound to 0/48 and 1/49. `baseline_slepc.py --bind core` now pins each rank to its
   block of the inherited cpuset. Verified per thread: all threads stay on 40/41.
5. **SLEPc's convergence test accepts pairs whose true residual exceeds tol.** It judges the
   Krylov residual estimate of the shift-inverted operator. baseline_slepc.py then computes each
   pair's true backward error with EPSComputeError, and SLEPc's own check rejects most of these
   pairs as well (report `success: false`). In 24 of the 25 trials of the 2-core tuning that failed
   tools/eval's backward-error test, SLEPc's own true backward error was already above tol
   (yee-pec-box-1e5 19 of 20, wg-square-1e5 3 of 3, interval 2 of 2). On yee-pec-box-1e5, 12 of
   these 19 were just above tol (1.21e-10), and the other 7 lay between 1.5e-10 and 1.3e-7.
   The norm of EPS_CONV_NORM is ||A||_inf (756.7 on yee-pec-box-1e5, against tools/eval's ||A||_2
   estimate 568.4). It makes SLEPc's test more lenient than tools/eval's by only that ratio,
   1.33x. The ratio decided 1 of the 25 trials (8.4e-11 by SLEPc's norm, 1.1e-10 by tools/eval's).
   Hence the accuracy group's `-eps_tol tol/10`, plus the `-eps_nev` padding for missed copies
   of multiple eigenvalues. The choice of norm matters little.
6. The first interval probe, yee-pec-box-1e5 [1, 200], has its lower end next to the 35475-fold
   zero eigenvalue. It was stopped after 3 minutes, before pitfall 3 had been found, so its cause
   is not separated. The tuned interval is [40, 200] (37 eigenvalues), an EM band query that keeps
   away from the zero cluster. [1, 200] was not retried.

## Pitfalls of the first version of this tooling (fixed after review)

1. **A 5-run confirmation cannot validate a randomly failing configuration.** The wg-square-1e5
   profile (`2x1/core`, BLR 1e-12) passed 5 of 5 confirmation runs. Later sessions measured 6 of 8
   and 4 of 6. The non-BLR profile it replaced screened 5th and was never confirmed in the same
   session. Fixed: the confirmation includes the start profile, and flaky cases get 20 runs per
   configuration (Method).
2. **compare.py's "ok" was the solver's own claim plus a set check against complete references
   only.** On the three cases with incomplete references nothing independent ran. The set check
   divided by max |lambda| of the whole spectrum. On syn-wide-range-real-1e4 (spectrum up to 1e6,
   wanted eigenvalues near 1) it accepted a run that returned the 11th-nearest eigenvalue instead
   of the 10th. Fixed: tools/eval judges every result. Set errors use tools/eval's yardstick
   |lambda-mu|/(|mu|+||A||/||B||) and are informational.
3. **compare.py expanded k-rules with the case config's k**, not with the k of a `--target`
   override. It could also credit a crashed rerun with the previous run's result: the result
   directory was not cleared, and the exit code was not checked. Both fixed.
4. **Class profiles** (the fallback for untuned cases) voted each knob separately. That dropped
   the accuracy padding one case needed, and it could combine a 1-rank layout with 2 slicing
   partitions, which SLEPc rejects. Fixed: options are voted among the cases with the voted
   layout and fitted to it, and accuracy options keep the strictest value. A class profile is
   still a heuristic that was never run as a whole: on yee-pec-cube-32 and yee-pec-box-1e4,
   neither the class profile nor the default gave a valid SLEPc run in the review's probes.
   compare.py marks such speedups "vs class profile". Interval profiles apply only to the
   interval they were tuned for.

## 16-core run for the lead (exclusive cores)

Run from the repository root after merging. The inner trial guards need a memory limit above the
shared-phase cap, so `HS_MAX_SHARED_MEM_GB=inf` (the outer guard is `--exclusive`). Estimated
1-1.5 h in total. wg-square-1e5 is flagged flaky (its 2-core profile records it), so its
confirmation runs 20 times per configuration. Peak memory per trial was at most 2.1 GiB at 2
cores. Every one of 16 ranks loads the whole case transiently, estimated at under 2 GiB extra.
compare.py evaluates in-process, so it runs under guard too. Give the `--exclusive` to the outer
guard only: a nested guard with `--exclusive` would wait forever for the lock its parent holds.

```bash
export HS_DATA_ROOT=$PWD/data NUMPY_MADVISE_HUGEPAGE=0 HS_MAX_SHARED_MEM_GB=inf
P16=24-30,33-41
GX="env/run.sh python3 tools/run/guard.py --exclusive --cores $P16 --mem-gb 24 --timeout 21600 --"
$GX python3 tools/bench/tune_slepc.py --cores $P16 --mem-gb 20 --tag tune-p16 --write-profiles \
  syn-mem27-46 yee-pec-box-1e5 wg-square-1e5 ss-qa8fk-qa8fm cav2d-rect-pert-1e5
$GX python3 tools/bench/tune_slepc.py --cores $P16 --mem-gb 20 --tag tune-p16 --write-profiles \
  --target '{"mode":"interval","lower":40,"upper":200}' yee-pec-box-1e5
env/run.sh python3 tools/bench/tune_slepc.py --report results/raw/tune-p16/*/tune.json
# head-to-head against the tuned 16-core profiles (interleaved, medians, tools/eval on every run):
$GX python3 tools/bench/compare.py --tag cmp-p16-best --cores $P16 --mem-gb 20 \
  --slepc-profile best --reps 5 --solvers hsolve,hsolve-nocert,slepc \
  syn-mem27-46 yee-pec-box-1e5 wg-square-1e5 ss-qa8fk-qa8fm cav2d-rect-pert-1e5
$GX python3 tools/bench/compare.py --tag cmp-p16-best-interval --cores $P16 --mem-gb 20 \
  --slepc-profile best --reps 5 --solvers hsolve,hsolve-nocert,slepc \
  --target '{"mode":"interval","lower":40,"upper":200}' yee-pec-box-1e5
```

At 16 cores the layout group covers 1x16, 2x8, 4x4, 8x2 and 16x1, each bound and unbound. The
ordering group adds parallel analysis with PT-SCOTCH/ParMETIS for multi-rank layouts, the most
promising knob against the 1.18 s serial analysis seen at 16 cores. Profiles are keyed by core
count: the 2-core entries stay, and the 16-core ones are added next to them. In the compare
tables a speedup is printed only when both solvers were valid in every repetition.

## Reproduce the 2-core numbers

```bash
export HS_CORES=40,41 HS_MAX_SHARED_MEM_GB=4 HS_DATA_ROOT=/home/zhlin/gencode/os2026/data NUMPY_MADVISE_HUGEPAGE=0
G="env/run.sh python3 tools/run/guard.py --skip-busy-check --cores 40,41 --mem-gb 4 --timeout 14400 --"
$G python3 tools/bench/tune_slepc.py --cores 40,41 --mem-gb 3.5 --tag tune-p2 --write-profiles \
   syn-mem27-46 yee-pec-box-1e5 wg-square-1e5 ss-qa8fk-qa8fm cav2d-rect-pert-1e5
$G python3 tools/bench/tune_slepc.py --cores 40,41 --mem-gb 3.5 --tag tune-p2 --write-profiles \
   --target '{"mode":"interval","lower":40,"upper":200}' yee-pec-box-1e5
$G python3 tools/bench/tune_slepc.py --cores 40,41 --mem-gb 3.5 --tag tune-p2-ext --write-profiles \
   --from-profile --groups blr,purify,prune syn-mem27-46 yee-pec-box-1e5 wg-square-1e5 ss-qa8fk-qa8fm cav2d-rect-pert-1e5
env/run.sh python3 tools/bench/tune_slepc.py --write-profiles --rescore results/raw/tune-p2-ext/*/tune.json \
   'results/raw/tune-p2/yee-pec-box-1e5@interval/tune.json'
# after the review: wg-square-1e5 with the 20-run confirmation (flaky flag set by the rescore above)
$G python3 tools/bench/tune_slepc.py --cores 40,41 --mem-gb 3.5 --tag tune-p2-fix --write-profiles \
   --from-profile --confirm-top 3 wg-square-1e5
$G python3 tools/bench/tune_slepc.py --cores 40,41 --mem-gb 3.5 --tag tune-p2-fix2 --write-profiles \
   --from-profile --groups prune --confirm-top 3 wg-square-1e5
$G python3 tools/bench/compare.py --tag cmp-p2-best-fix --cores 40,41 --mem-gb 3.5 --slepc-profile best \
   --reps 5 --solvers hsolve,hsolve-nocert,slepc syn-mem27-46 yee-pec-box-1e5 wg-square-1e5 ss-qa8fk-qa8fm cav2d-rect-pert-1e5
$G python3 tools/bench/compare.py --tag cmp-p2-best-fix-interval --cores 40,41 --mem-gb 3.5 --slepc-profile best \
   --reps 3 --solvers hsolve,hsolve-nocert,slepc --target '{"mode":"interval","lower":40,"upper":200}' yee-pec-box-1e5
```
