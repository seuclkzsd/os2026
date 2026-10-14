# cmp-p9-1e6-krylov: SLEPc's Krylov group at 1e6 (P9 "mpi" review fix)

`tools/bench/tune_slepc.py --groups ncv,restart` on the two 3-D 1e6 cases (c005, 2026-10-14
14:12-14:59, measurement slot cores 16-31, one guard call per case, `--mem-gb 32`; every trial
evaluated by tools/eval against the complete reference). The search starts from the P9 profiles of
`results/p9_mpi.md` section 2.1 (syn-mem27-100: `1x16`, SLEPc and MUMPS defaults; yee-pec-box-1e6:
`--from-profile`, the 4 x 4 BLR / ParMETIS profile), screens `-eps_ncv` k+8 / 2k / 3k / 4k
(SLEPc's default is max(2 nev, nev + 15) = 27 for nev = 12 on both cases) and then
`-eps_krylovschur_restart` 0.3 / 0.7 on the incumbent, one run each, and confirms the default, the
start, the best and the best runner-up with 2 interleaved runs each (medians). Solver time =
`-log_view` setup + solve + vectors, the scope of compare.py's SLEPc time. The winners are the
profiles of `results/cmp-p9-1e6-profiles.json`.

| case | type | target | P | default (s) | best (s) | speedup | best configuration | analysis | numeric (#) | solves (#/s) | ortho | valid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| syn-mem27-100 | HEP | k=12 | 16 | 124.827 | 115.628 | 1.08x | `1x16 eps_ncv=3k` | 14.245 | 23.231 (1) | 105 / 75.778 | 2.007 | yes |
| yee-pec-box-1e6 | GHEP | k=10 | 16 | 73.882 (invalid) | 59.525 | 1.24x | `4x4/core eps_ncv=3k eps_nev=k+2 st_mat_mumps_cntl_7=1e-10 st_mat_mumps_icntl_28=2 st_mat_mumps_icntl_29=2 st_mat_mumps_icntl_35=2` | 6.793 | 24.723 (1) | 59 / 26.803 | 0.747 | yes |

(Rendered by `tune_slepc.py --report`; "default" is the untuned `1x16`. On yee-pec-box-1e6 the
start profile `B` below is the reference: 68.821 s.)

Confirmation (interleaved; median [min-max] solver time, s):

- **syn-mem27-100**: `1x16` 124.827 [124.693-124.962]; `1x16 eps_ncv=3k` **115.628**
  [115.572-115.684]; `1x16 eps_krylovschur_restart=0.7 eps_ncv=3k` 118.840 [118.803-118.876].
- **yee-pec-box-1e6**: `1x16` 73.882 [72.850-74.913] INVALID (1 of the 10 eigenvalues missing);
  `B` 68.821 [68.298-69.345]; `B eps_ncv=3k` **59.525** [59.007-60.042];
  `B eps_krylovschur_restart=0.7 eps_ncv=3k` 63.821 [63.706-63.937].

## Every trial

`B` = `4x4/core eps_nev=k+2 st_mat_mumps_cntl_7=1e-10 st_mat_mumps_icntl_28=2
st_mat_mumps_icntl_29=2 st_mat_mumps_icntl_35=2` (the P9 yee profile: BLR at 1e-10, ParMETIS).

**syn-mem27-100** (n = 1e6, k = 12)

| # | phase | configuration | solver s | analysis | numeric | solves (#/s) | valid |
|---|---|---|---|---|---|---|---|
| 0 | default | `1x16` | 124.65 | 14.20 | 23.25 | 118 / 85.10 | yes |
| 1 | ncv | `1x16 eps_ncv=k+8` | 141.97 | 14.27 | 23.16 | 142 / 102.48 | yes |
| 2 | ncv | `1x16 eps_ncv=2k` | 121.79 | 14.31 | 23.17 | 114 / 82.41 | yes |
| 3 | ncv | `1x16 eps_ncv=3k` | 115.87 | 14.15 | 23.54 | 105 / 75.75 | yes |
| 4 | ncv | `1x16 eps_ncv=4k` | 125.43 | 14.25 | 23.20 | 117 / 84.66 | yes |
| 5 | restart | `1x16 eps_krylovschur_restart=0.3 eps_ncv=3k` | 120.39 | 14.15 | 23.26 | 112 / 80.76 | yes |
| 6 | restart | `1x16 eps_krylovschur_restart=0.7 eps_ncv=3k` | 119.15 | 14.21 | 23.26 | 109 / 78.80 | yes |
| 7 | confirm0 | `1x16` | 124.96 | 14.32 | 23.29 | 118 / 85.24 | yes |
| 8 | confirm0 | `1x16 eps_ncv=3k` | 115.68 | 14.30 | 23.22 | 105 / 75.77 | yes |
| 9 | confirm0 | `1x16 eps_krylovschur_restart=0.7 eps_ncv=3k` | 118.80 | 14.33 | 23.19 | 109 / 78.53 | yes |
| 10 | confirm1 | `1x16 eps_ncv=3k` | 115.57 | 14.19 | 23.24 | 105 / 75.78 | yes |
| 11 | confirm1 | `1x16 eps_krylovschur_restart=0.7 eps_ncv=3k` | 118.88 | 14.33 | 23.15 | 109 / 78.59 | yes |
| 12 | confirm1 | `1x16` | 124.69 | 14.17 | 23.25 | 118 / 85.11 | yes |

**yee-pec-box-1e6** (n = 1027733, k = 10)

| # | phase | configuration | solver s | analysis | numeric | solves (#/s) | valid |
|---|---|---|---|---|---|---|---|
| 0 | default | `1x16` | 74.82 | 13.42 | 24.06 | 50 / 36.15 | no: 1 eigenvalue missing |
| 1 | start | `B` | 69.22 | 6.98 | 25.10 | 78 / 35.45 | yes |
| 2 | ncv | `B eps_ncv=k+8` | 60.26 | 6.91 | 25.25 | 60 / 27.00 | no: 1 of 10 pairs eta > tol |
| 3 | ncv | `B eps_ncv=2k` | 63.56 | 6.87 | 24.91 | 68 / 30.52 | yes |
| 4 | ncv | `B eps_ncv=3k` | 58.15 | 6.74 | 24.35 | 57 / 25.88 | yes |
| 5 | ncv | `B eps_ncv=4k` | 59.71 | 6.83 | 25.20 | 58 / 26.38 | no: 1 of 10 pairs eta > tol |
| 6 | restart | `B eps_krylovschur_restart=0.3 eps_ncv=3k` | 68.10 | 6.94 | 24.91 | 77 / 34.60 | yes |
| 7 | restart | `B eps_krylovschur_restart=0.7 eps_ncv=3k` | 63.14 | 6.73 | 24.48 | 67 / 30.43 | yes |
| 8 | confirm0 | `1x16` | 74.91 | 13.39 | 23.98 | 50 / 36.31 | no: 1 eigenvalue missing |
| 9 | confirm0 | `B` | 69.34 | 6.85 | 25.30 | 78 / 35.51 | yes |
| 10 | confirm0 | `B eps_ncv=3k` | 60.04 | 6.84 | 25.17 | 59 / 26.83 | yes |
| 11 | confirm0 | `B eps_krylovschur_restart=0.7 eps_ncv=3k` | 63.71 | 6.97 | 24.98 | 67 / 30.24 | yes |
| 12 | confirm1 | `B` | 68.30 | 6.74 | 24.43 | 78 / 35.42 | yes |
| 13 | confirm1 | `B eps_ncv=3k` | 59.01 | 6.75 | 24.28 | 59 / 26.77 | yes |
| 14 | confirm1 | `B eps_krylovschur_restart=0.7 eps_ncv=3k` | 63.94 | 6.84 | 25.59 | 66 / 30.05 | yes |
| 15 | confirm1 | `1x16` | 72.85 | 13.54 | 24.02 | 47 / 34.09 | no: 1 eigenvalue missing |

* The whole gain is fewer single-RHS solves (syn 118 -> 105, yee 78 -> 57-59); analysis and
  numeric factorization are unchanged. A larger subspace (3k = 36 / 30 vectors) needs fewer
  restarts; 4k costs more orthogonalization and on syn more solves again.
* With BLR factors (yee) the ncv values k+8 and 4k each returned one pair above the tolerance:
  the BLR-perturbed operator makes SLEPc's accuracy depend on the Krylov parameters.
  `-eps_mpd` was not screened: with nev = 12 the projected problem (dimension ncv <= 48) and the
  orthogonalization cost 0.7-2 s of 60-125 s, which is all that mpd < ncv could save.

Commands (from the work tree on c005, `D=$HOME/p7_p7-membudget/data`):

```bash
python3 tools/run/guard.py --exclusive --timeout 2700 --mem-gb 32 -- \
  python3 tools/bench/tune_slepc.py syn-mem27-100 --cores 16-31 --tag p9fix-krylov --root $D \
    --mem-gb 32 --timeout 900 --groups ncv,restart --screen-reps 1 --confirm-reps 2 --confirm-top 1 \
    --flaky-confirm-reps 2 --profiles-file results/cmp-p9-1e6-profiles.json   # the P9 profiles (fd27a13)
# yee-pec-box-1e6: the same with --from-profile
python3 tools/bench/tune_slepc.py --report results/raw/p9fix-krylov/*/tune.json
```
