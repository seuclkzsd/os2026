# SLEPc baseline tuning (P4, module baseline)

Goal: make the comparison target as strong as possible, so that a speedup claim of hsolve is a claim
against the fastest **valid** open-source configuration at equal core count: SLEPc 3.26 Krylov-Schur +
shift-and-invert + MUMPS 5.8.2 (env `os2026-ref`, ParMETIS / PT-SCOTCH available).

Measured 2026-10-11 on 2 cores (40,41) of the shared machine (load average 40-44 on 96 hardware
threads). This is noisy: the same configuration's median moved by up to 30 % between runs an hour
apart (ss-qa8fk-qa8fm default: 4.80 s, later 3.52 s). That is why every decision rests on
interleaved repetitions and medians, never on single runs.

## Bottom line

* Tuned SLEPc vs SLEPc's default (2 cores, interleaved medians): **1.13x** syn-mem27-46, **1.20x**
  ss-qa8fk-qa8fm, **1.23x** wg-square-1e5, **1.94x** cav2d-rect-pert-1e5, **1.03x** (noise)
  for the interval case.
* SLEPc's default is **not valid** on yee-pec-box-1e5 (all 5 confirmation runs) and on
  wg-square-1e5 (1 of 5 runs). It misses a copy of a multiple eigenvalue and leaves pairs above
  tol. The cheapest valid configuration on yee-pec-box-1e5 costs as much as the invalid default
  (0.98x).
* **hsolve vs the tuned baseline at 2 cores** (same scope; table below): hsolve (certificate on)
  runs at 0.53x / 0.68x / 0.56x / 0.73x of SLEPc's speed on syn-mem27-46 / yee-pec-box-1e5 /
  ss-qa8fk-qa8fm / cav2d-rect-pert-1e5, so it is *slower* there. It is 1.20x faster on wg-square-1e5
  and 1.45x faster on the interval case. With the certificate off it reaches 0.78x / 1.00x / 0.73x
  / 0.92x / 1.14x. The bar moved: syn-mem27-46 measured 5.1 s untuned at 2 cores before and now
  4.7-5.3 s tuned, depending on the machine's load. The gap comes from hsolve's analysis and its
  2 extra certificate factorizations (see "Lessons for hsolve").

## Method (tools/bench/tune_slepc.py)

* Configuration = ranks x threads layout + binding + PETSc options (`tools/bench/slepc_profile.py`;
  MUMPS controls carry the spectral transform's prefix: `-st_mat_mumps_icntl_7`). Option values
  may be k-rules (`2k`, `k+4`), so a profile serves any k.
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
  conservative 2-norm estimates, completeness against the complete references, duplicates). The
  convergence test stays EPS_CONV_NORM at the case tol. Only the accuracy group may tighten it.
  Options PETSc never used make a trial invalid (a misspelt option would silently tune nothing).
* Metric: the solver time from `-log_view`: EPSSetUp (MUMPS analysis + factorization) + EPSSolve
  + the eigenvector computation EPSSolve defers (purification). It has the same scope as hsolve's
  `time_total_s`. Python/MPI start-up, case loading and output are excluded
  (`tools/bench/slepc_log.py`).
* Confirmation: default, best and 2 runners-up, 5 interleaved repetitions (rotating order). Medians
  decide, and a configuration must be valid in **every** repetition.
* The 2-core run: `tune-p2` (all groups except BLR/prune for the nearest cases), then
  `tune-p2-ext` (`--from-profile --groups blr,purify,prune`), then `--rescore` with the final
  selection rule. Raw trials: `results/raw/tune-p2*/<case>/` (not tracked). Profiles:
  `tools/bench/slepc_profiles.json`.

## Best configuration per case (2 cores)

Solver time in s, median of 5 interleaved runs. Phases (analysis, numeric, solves, ortho) are the
best configuration's medians.

| case | type | target | P | default (s) | best (s) | speedup | best configuration | analysis | numeric (#) | solves (#/s) | ortho | valid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| syn-mem27-46 | HEP | k=12 | 2 | 5.285 | 4.675 | 1.13x | `1x2 eps_ncv=2k st_mat_mumps_icntl_7=4` | 0.498 | 1.822 (1) | 36 / 2.290 | 0.050 | yes |
| yee-pec-box-1e5 | GHEP | k=10 | 2 | 7.194 (invalid) | 7.337 | 0.98x | `1x2 eps_nev=k+4 eps_purify=0` | 1.241 | 1.857 (1) | 50 / 4.097 | 0.168 | yes |
| wg-square-1e5 | GHIEP | k=18 | 2 | 2.826 (invalid) | 2.290 | 1.23x | `2x1/core st_mat_mumps_cntl_7=1e-12 st_mat_mumps_icntl_13=1 st_mat_mumps_icntl_35=2 st_mat_mumps_icntl_7=6` | 0.113 | 0.132 (1) | 80 / 1.507 | 0.357 | yes |
| ss-qa8fk-qa8fm | GHEP | k=10 | 2 | 3.523 | 2.940 | 1.20x | `1x2 eps_ncv=3k eps_purify=0 st_mat_mumps_icntl_7=4` | 0.556 | 0.661 (1) | 44 / 1.448 | 0.166 | yes |
| cav2d-rect-pert-1e5 | GHEP | k=6 | 2 | 1.104 | 0.570 | 1.94x | `2x1/core eps_purify=0 st_mat_mumps_icntl_7=2` | 0.051 | 0.077 (1) | 22 / 0.377 | 0.037 | yes |
| yee-pec-box-1e5 | GHEP | [40, 200] (37 eigenvalues) | 2 | 23.800 | 23.047 | 1.03x | `2x1 eps_krylovschur_nev=60` | 1.286 | 4.038 (2) | 194 / 15.591 | 1.818 | yes |

(`1x2` = 1 MPI rank x 2 OpenMP threads; `/core` = ranks pinned to disjoint cores of the guard's
cpuset and OpenMP threads to cores. ICNTL(7): 2 AMF, 4 PORD, 6 QAMD; default 7 = automatic, which
picked METIS here.)

Confirmation runs (median [min-max] s; interval case from `tune-p2`, the others from `tune-p2-ext`):

- **syn-mem27-46**: default 5.285 [5.127-6.012]; best 4.675 [4.493-5.219]; + BLR 1e-10 4.904; + BLR 1e-12 5.059
- **yee-pec-box-1e5**: default 7.194 [7.076-7.625] INVALID in 5/5 (1 pair above tol, 1 eigenvalue copy missing);
  `eps_nev=k+4 eps_purify=0` 7.337 [6.840-8.626]; + BLR 1e-12 8.115; `eps_nev=k+4` (purification on) 9.179
- **wg-square-1e5**: default 2.826 [2.597-3.696] INVALID in 1/5 (1 eigenvalue missing); best 2.290 [1.874-2.451];
  BLR 1e-10 2.477; BLR 1e-10 without ICNTL(13)=1 INVALID in 1/5 (5 missing). In `tune-p2`, the screening winner
  `2x1/core icntl_7=6` was also INVALID in 1/5 (6 missing): SLEPc's indefinite (GHIEP) Krylov-Schur is flaky here.
- **ss-qa8fk-qa8fm**: default 3.523 [3.351-3.704]; best 2.940 [2.867-3.293]; + ICNTL(58)=1 2.935 (same, more options)
- **cav2d-rect-pert-1e5**: default 1.104 [1.081-1.273]; best 0.570 [0.556-0.643]; + BLR 1e-10 0.559 (within noise)
- **yee-pec-box-1e5 [40, 200]**: default 23.800 [21.917-26.184]; best 23.047 [20.496-29.535]; `1x2 nev=60`
  26.565; `2x1/core nev=60` 27.441. All within the noise.

## What the knobs do (screening, 2 cores)

| knob | effect | evidence (screening trials, solver s) |
|---|---|---|
| MUMPS ordering, 2-D cases | METIS (the automatic choice) costs 0.5-0.7 s of analysis. QAMD/AMF cost 0.06-0.1 s with equal numeric time: **the largest single gain** | wg-square-1e5 analysis 0.72 -> 0.09 s (QAMD); cav2d analysis 0.46 -> 0.06 s (AMF) |
| MUMPS ordering, 3-D cases | PORD halves the analysis at similar factor cost. AMD/AMF/QAMD are 2-5x worse in numeric and solve | syn-mem27-46 analysis 1.08 -> 0.49 s, total 4.94 -> 4.70; yee-pec-box-1e5 PORD is slower (numeric 2.6 vs 1.6 s) |
| `-eps_purify 0` (generalized) | saves about k solves (purification applies OP to every vector). Valid on all GHEP cases, invalid on GHIEP (wg-square: 3 pairs above tol) | yee-pec-box 68 -> 50 solves; cav2d 29 -> 22; ss-qa8fk 54 -> 44 |
| `-eps_ncv` | 2k-3k beat the default max(2k, k+15) on some cases, 4k always loses | syn-mem27-46 2k: 41 -> 36 solves; ss-qa8fk 3k: 58 -> 54 |
| layout at 2 cores | 2x1 (MPI) wins on the 2-D cases, where solves and vector operations dominate; 1x2 wins on 3-D | cav2d 1.50 -> 1.19 s; syn-mem27-46 2x1 5.27 vs 1x2 4.94 |
| parallel analysis ICNTL(28)=2 | slower than the best sequential ordering at 2 ranks (PT-SCOTCH 0.19-0.42 s vs QAMD/AMF 0.06-0.09 s) | worth re-testing at 16 ranks, where the analysis is serial and 1.18 s |
| BLR ICNTL(35)=2 | no gain at n = 1e5 (higher numeric, slightly fewer flops in the solves). CNTL(7)=1e-10 breaks accuracy on yee-pec-box | syn-mem27-46 4.68 -> 4.90/5.06 |
| ICNTL(48)=0, ICNTL(8)=0, ICNTL(58)=1, restart 0.3/0.7, GNHEP for GHIEP | no gain, or invalid (restart 0.7 and GNHEP on wg-square: missing eigenvalues / duplicate vector) | negative results kept out of the profiles |
| `-eps_krylovschur_nev` (slicing) | 10/20 per shift are **invalid** (6 of 37 pairs above tol). 60 is valid, no faster than the default 40 | interval case |

## Head-to-head at 2 cores (tools/bench/compare.py --slepc-profile best --reps 3)

Same scope: hsolve `time_total_s` against SLEPc's `-log_view` setup + solve + vectors. Medians of 3
interleaved runs. Every result passed the eigenvalue-set check against the complete references
(`ok` = success and correct set).

| case | hsolve (s) | hsolve-nocert (s) | SLEPc tuned (s) | hsolve speedup | nocert speedup | hsolve analysis / numeric (#) / solves | SLEPc analysis / numeric (#) / solves |
|---|---|---|---|---|---|---|---|
| syn-mem27-46 | 9.96 | 6.74 | 5.29 | 0.53x | 0.78x | 1.45 / 5.32 (3) / 36 in 2.39 s | 0.52 / 1.97 (1) / 36 in 2.56 s |
| yee-pec-box-1e5 | 9.52 | 6.49 | 6.47 | 0.68x | 1.00x | 1.36 / 4.77 (3) / 50 in 3.16 s | 1.22 / 1.67 (1) / 50 in 3.44 s |
| wg-square-1e5 | 1.93 | 2.03 | 2.31 | 1.20x | 1.14x | 0.05 / 0.32 (2) / 57 in 1.08 s | 0.10 / 0.13 (1) / 77 in 1.53 s |
| ss-qa8fk-qa8fm | 6.37 | 4.94 | 3.59 | 0.56x | 0.73x | 1.01 / 3.11 (4) / 50 in 1.73 s | 0.59 / 0.78 (1) / 44 in 1.93 s |
| cav2d-rect-pert-1e5 | 0.84 | 0.66 | 0.61 | 0.73x | 0.92x | 0.04 / 0.34 (4) / 22 in 0.34 s | 0.06 / 0.08 (1) / 22 in 0.40 s |
| yee-pec-box-1e5 [40, 200] | 15.94 | 16.49 | 23.17 | 1.45x | 1.41x | 1.32 / 5.06 (3) / 113 in 8.07 s | 1.25 / 3.50 (2) / 239 in 15.91 s |

Wall time including start-up and loading (guard) still favours hsolve on every case: SLEPc's
Python + MPI start-up + per-rank case loading costs 2.3-2.5 s. That is not solver time, and claims
must use the solver-time column.

Lessons for hsolve (input for the analysis / certfast modules, not changed here):
1. PORD analysis on 3-D cases: SLEPc's 0.50 s vs hsolve's 1.45 s analysis on syn-mem27-46, with
   the same single-factorization cost (1.8-2.0 s).
2. The certificate's extra factorizations are the gap: hsolve-nocert is at parity on
   yee-pec-box-1e5 and within 0.78-0.92x elsewhere. hsolve's numeric time is 2.7-4x SLEPc's
   because of 3-4 factorizations.
3. hsolve's solve phase is already at parity or better (same or fewer solves, similar time per solve).
4. Equal accuracy: every run of both solvers in this table returned the correct eigenvalue set.
   SLEPc's configurations are the ones that passed tools/eval in every confirmation run.

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
5. SLEPc's EPS_CONV_NORM uses ||.||_inf (`-eps_view`: norm(A) = 756.701 = ||A||_inf of
   yee-pec-box-1e5). That is an upper bound of the 2-norm, so SLEPc's backward error is the more
   lenient one, and its "converged" pairs can fail tools/eval at the same tol.
6. The first interval probe, yee-pec-box-1e5 [1, 200], has its lower end next to the 35475-fold
   zero eigenvalue. It was stopped after 3 minutes, before pitfall 3 had been found, so its cause
   is not separated. The tuned interval is [40, 200] (37 eigenvalues), an EM band query that keeps
   away from the zero cluster. [1, 200] was not retried.

## 16-core run for the lead (exclusive cores)

Run from the repository root after merging. The inner trial guards need a memory limit above the
shared-phase cap, so `HS_MAX_SHARED_MEM_GB=inf` (the outer guard is `--exclusive`). Estimated about
1 h in total. Peak memory per trial was at most 2.1 GiB at 2 cores. Every one of 16 ranks loads
the whole case transiently, estimated at under 2 GiB extra.

```bash
export HS_DATA_ROOT=$PWD/data NUMPY_MADVISE_HUGEPAGE=0 HS_MAX_SHARED_MEM_GB=inf
P16=24-30,33-41
env/run.sh python3 tools/run/guard.py --exclusive --cores $P16 --mem-gb 24 --timeout 21600 -- \
  python3 tools/bench/tune_slepc.py --cores $P16 --mem-gb 20 --tag tune-p16 --write-profiles \
  syn-mem27-46 yee-pec-box-1e5 wg-square-1e5 ss-qa8fk-qa8fm cav2d-rect-pert-1e5
env/run.sh python3 tools/run/guard.py --exclusive --cores $P16 --mem-gb 24 --timeout 21600 -- \
  python3 tools/bench/tune_slepc.py --cores $P16 --mem-gb 20 --tag tune-p16 --write-profiles \
  --target '{"mode":"interval","lower":40,"upper":200}' yee-pec-box-1e5
env/run.sh python3 tools/bench/tune_slepc.py --report results/raw/tune-p16/*/tune.json
# head-to-head against the tuned 16-core profiles (interleaved, medians):
env/run.sh python3 tools/bench/compare.py --tag cmp-p16-best --cores $P16 --exclusive --mem-gb 24 \
  --slepc-profile best --reps 5 --solvers hsolve,hsolve-nocert,slepc \
  syn-mem27-46 yee-pec-box-1e5 wg-square-1e5 ss-qa8fk-qa8fm cav2d-rect-pert-1e5
env/run.sh python3 tools/bench/compare.py --tag cmp-p16-best-interval --cores $P16 --exclusive --mem-gb 24 \
  --slepc-profile best --reps 5 --solvers hsolve,hsolve-nocert,slepc \
  --target '{"mode":"interval","lower":40,"upper":200}' yee-pec-box-1e5
```

At 16 cores the layout group covers 1x16, 2x8, 4x4, 8x2 and 16x1, each bound and unbound. The
ordering group adds parallel analysis with PT-SCOTCH/ParMETIS for multi-rank layouts, the most
promising knob against the 1.18 s serial analysis seen at 16 cores. Profiles are keyed by core
count: the 2-core entries stay, and the 16-core ones are added next to them.

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
env/run.sh python3 tools/bench/compare.py --tag cmp-p2-best --cores 40,41 --mem-gb 3.5 --slepc-profile best \
   --reps 3 --solvers hsolve,hsolve-nocert,slepc syn-mem27-46 yee-pec-box-1e5 wg-square-1e5 ss-qa8fk-qa8fm cav2d-rect-pert-1e5
```
