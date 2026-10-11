# Evaluation of cmp4b (hsolve vs SLEPc)

Run `/home/zhlin/gencode/os2026/results/raw/cmp4b`, data root `/home/zhlin/gencode/os2026/data`; 30 evaluation(s) by `python -m tools.eval.sweep` (tools/eval/sweep.py), evaluation JSONs in `results/raw/eval/cmp4b/<case>/<solver>/evaluation.json`.

Verdicts - **hsolve**: 15 pass / 0 fail; **slepc**: 10 pass / 5 fail, 1 overclaim(s)

Metrics (tools/eval/evaluate.py, recomputed from the case matrices and the returned vectors): ref = reference kind used (discrete-exact / analytic / dense: certified complete; solver: ARPACK shift-invert, uncertified; pooled: accepted pairs of all solvers of this run, uncertified; n/c = not comparable, e.g. continuum-model values; - = none); time = solver-reported `time_total_s`; wall / peak = wall time and peak RSS of the whole process tree measured by tools/run/guard.py (peak marked * is solver-reported, when no guard summary exists); eta = backward error ||Ax - lambda Bx||_2 / ((||A||_2 + |lambda| ||B||_2) ||x||_2); rel res = ||Ax - lambda Bx||_2 / (|lambda| ||Bx||_2), which stays informative where |lambda| ||B||_2 ||x||_2 dwarfs the residual and eta tells little (e.g. a badly conditioned B); eig err = |lambda - mu|/|mu| / |lambda - mu|/(|mu| + ||A||_2/||B||_2) against the matched reference value (not against a pooled reference, which holds the solvers' own values); angle = largest principal angle between computed and reference eigenspaces per eigenvalue cluster (needs reference eigenvectors); B-orth = ||X^H B X - I||_2 (HEP/GHEP; not part of the verdict); completeness = status (found/expected, + pairs outside the target set): complete/incomplete against a certified reference, consistent/inconsistent against an uncertified one; claim = the solver's own success flag vs the verdict (over = success claimed but the evaluation fails, under = failure reported but the evaluation passes). Verdict pass = every eta <= tol, no duplicate vectors, completeness not incomplete/inconsistent, no spurious eigenvalue.

| case | solver | problem | n | k | ref | time [s] | wall [s] | peak [MiB] | max eta | max rel res | max eig err (rel / scaled) | max angle | B-orth | completeness | claim | verdict |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| cav2d-rect-pert-1e5 | hsolve | GHEP | 99459 | 6 | solver | 1.04 | 1.17 | 125 | 5.4e-15 | 1.5e-09 | 1.5e-13 / 2.1e-18 | 1.3e-11 | 1.8e-15 | consistent (6/6) | ok | pass |
| cav2d-rect-pert-1e5 | slepc | GHEP | 99459 | 6 | solver | 1.04 | 3.28 | 252 | 8.8e-17 | 1.6e-11 | 9.9e-13 / 1.4e-17 | 1.4e-11 | 1.3e-12 | consistent (6/6) | ok | pass |
| ss-2cubes_sphere | hsolve | HEP | 101492 | 10 | pooled | 6.25 | 6.45 | 689 | 6.3e-15 | 6.3e-13 | - / - | - | 1.6e-15 | consistent (10/10) | ok | pass |
| ss-2cubes_sphere | slepc | HEP | 101492 | 10 | pooled | 4.43 | 6.74 | 772 | 1.3e-05 | 1.3e-03 | - / - | - | 3.4e-15 | consistent (10/10) | ok | **fail** |
| ss-helm3d01 | hsolve | HEP | 32226 | 10 | solver | 0.936 | 1.01 | 148 | 5.2e-13 | 2.6e-08 | 7.9e-13 / 1.7e-17 | 9.4e-12 | 1.7e-15 | consistent (10/10) | ok | pass |
| ss-helm3d01 | slepc | HEP | 32226 | 10 | solver | 0.801 | 2.93 | 244 | 3.9e-13 | 4.3e-08 | 7.9e-11 / 3.2e-16 | 9.1e-12 | 4.8e-15 | consistent (10/10) | ok | pass |
| ss-qa8fk-qa8fm | hsolve | GHEP | 66127 | 10 | solver | 3.53 | 3.73 | 475 | 1.7e-13 | 6.6e-12 | 9.5e-15 / 2.7e-16 | 8.7e-12 | 2.1e-15 | consistent (10/10) | ok | pass |
| ss-qa8fk-qa8fm | slepc | GHEP | 66127 | 10 | solver | 3.31 | 5.54 | 570 | 1.8e-13 | 7.0e-12 | 1.3e-13 / 3.7e-15 | 2.9e-10 | 8.9e-12 | consistent (10/10) | ok | pass |
| ss-tem27623 | hsolve | GHEP | 27623 | 10 | solver | 0.902 | 1.01 | 140 | 1.5e-21 | 4.1e-13 | 9.3e-16 / 9.3e-16 | 1.0e-12 | 1.7e-15 | consistent (10/10) | ok | pass |
| ss-tem27623 | slepc | GHEP | 27623 | 10 | solver | 0.806 | 3.02 | 239 | 2.2e-13 | 6.8e-05 | 4.2e-07 / 4.2e-07 | 5.1e-03 | 6.8e-07 | consistent (10/10) | ok | pass |
| syn-ghep-near-singular-real-1e5 | hsolve | GHEP | 100000 | 10 | discrete-exact | 1.79 | 2.21 | 203 | 2.0e-15 | 2.2e-13 | 1.7e-15 / 5.0e-16 | - | 1.2e-15 | complete (10/10) | ok | pass |
| syn-ghep-near-singular-real-1e5 | slepc | GHEP | 100000 | 10 | discrete-exact | 2.02 | 4.22 | 367 | 1.4e-16 | 8.4e-15 | 3.1e-15 / 9.5e-16 | - | 3.5e-11 | complete (10/10) | ok | pass |
| syn-lap3d-aniso-1e5 | hsolve | HEP | 97200 | 12 | discrete-exact | 4.27 | 4.38 | 525 | 9.5e-13 | 5.1e-07 | 2.0e-11 / 5.2e-17 | - | 2.3e-15 | complete (12/12) | ok | pass |
| syn-lap3d-aniso-1e5 | slepc | HEP | 97200 | 12 | discrete-exact | 3.28 | 5.42 | 550 | 9.3e-13 | 4.7e-07 | 2.3e-10 / 3.0e-16 | - | 4.6e-15 | complete (12/12) | ok | pass |
| syn-mem27-46 | hsolve | HEP | 97336 | 12 | discrete-exact | 5.12 | 5.28 | 749 | 4.8e-13 | 4.3e-07 | 1.4e-11 / 1.7e-17 | - | 1.9e-15 | complete (12/12) | ok | pass |
| syn-mem27-46 | slepc | HEP | 97336 | 12 | discrete-exact | 3.82 | 6.04 | 863 | 4.9e-13 | 4.0e-07 | 4.0e-10 / 3.9e-16 | - | 4.4e-15 | complete (12/12) | ok | pass |
| syn-zero-cluster-real-1e5 | hsolve | HEP | 100000 | 21 | discrete-exact | 3.97 | 4.23 | 350 | 4.2e-15 | 6.6e-06 | 2.5e-08 / 6.7e-17 | - | 2.4e-15 | complete (21/21) | ok | pass |
| syn-zero-cluster-real-1e5 | slepc | HEP | 100000 | 21 | discrete-exact | 2.07 | 4.25 | 352 | 3.4e-15 | 7.3e-07 | 3.3e-07 / 1.7e-15 | - | 4.9e-15 | complete (21/21) | ok | pass |
| wg-halfslab-1e5 | hsolve | GHIEP | 99225 | 28 | solver | 2.73 | 2.84 | 360 | 1.0e-14 | 2.5e-08 | 1.6e-10 / 4.5e-15 | 2.2e-09 | - | consistent (28/28) | ok | pass |
| wg-halfslab-1e5 | slepc | GHIEP | 99225 | 28 | solver | 2.94 | 5.22 | 358 | 1.8e-10 | 2.5e-05 | 2.7e-09 / 9.2e-14 | 9.8e-06 | - | consistent (28/28) | ok | **fail** |
| wg-square-1e5 | hsolve | GHIEP | 99225 | 18 | solver | 2.11 | 3.38 | 273 | 6.4e-16 | 1.6e-10 | 6.3e-10 / 2.1e-13 | 5.9e-08 | - | consistent (18/18) | ok | pass |
| wg-square-1e5 | slepc | GHIEP | 99225 | 18 | solver | 2.31 | 4.53 | 333 | 4.4e-12 | 2.0e-07 | 6.4e-10 / 2.1e-13 | 2.2e-07 | - | consistent (18/18) | ok | pass |
| wg-square-pert-1e4 | hsolve | GHIEP | 9801 | 18 | solver | 0.115 | 0.167 | 41 | 5.7e-13 | 4.5e-09 | 3.3e-13 / 8.3e-16 | 1.5e-11 | - | consistent (18/18) | ok | pass |
| wg-square-pert-1e4 | slepc | GHIEP | 9801 | 18 | solver | 0.155 | 2.61 | 146 | 7.1e-15 | 5.7e-11 | 3.4e-12 / 1.3e-14 | 4.8e-10 | - | consistent (18/18) | ok | pass |
| yee-pec-box-1e4 | hsolve | GHEP | 10064 | 11 | discrete-exact | 0.257 | 0.317 | 58 | 6.7e-13 | 4.3e-11 | 1.2e-15 / 1.3e-17 | - | 2.4e-15 | complete (11/11) | ok | pass |
| yee-pec-box-1e4 | slepc | GHEP | 10064 | 11 | discrete-exact | 0.464 | 2.68 | 170 | 5.4e-12 | 5.5e-10 | 9.4e-14 / 5.0e-16 | - | 1.8e-12 | incomplete (10/11, +1 extra) | **over** | **fail** |
| yee-pec-box-1e5 | hsolve | GHEP | 109744 | 10 | discrete-exact | 6.18 | 6.33 | 794 | 5.9e-14 | 5.2e-11 | 2.1e-14 / 2.9e-17 | - | 1.5e-15 | complete (10/10) | ok | pass |
| yee-pec-box-1e5 | slepc | GHEP | 109744 | 10 | discrete-exact | 4.36 | 6.63 | 811 | 1.6e-10 | 4.8e-08 | 5.9e-13 / 6.8e-16 | - | 2.8e-09 | incomplete (9/10, +1 extra) | ok | **fail** |
| yee-pec-cube-32 | hsolve | HEP | 92256 | 12 | discrete-exact | 6.16 | 6.28 | 649 | 5.8e-14 | 1.2e-11 | 2.2e-15 / 8.7e-18 | - | 1.9e-15 | complete (12/12) | ok | pass |
| yee-pec-cube-32 | slepc | HEP | 92256 | 12 | discrete-exact | 4 | 6.22 | 733 | 2.2e-07 | 5.5e-05 | 5.9e-13 / 1.4e-15 | - | 3.7e-15 | incomplete (11/12, +1 extra) | ok | **fail** |

## Failures

- ss-2cubes_sphere / slepc: 5 of 10 pairs have backward error > tol=1e-10
- wg-halfslab-1e5 / slepc: 2 of 28 pairs have backward error > tol=1e-10
- yee-pec-box-1e4 / slepc: completeness incomplete: 1 missing, boundary deficit 0, 11 distinct returned
- yee-pec-box-1e5 / slepc: 1 of 10 pairs have backward error > tol=1e-10; completeness incomplete: 1 missing, boundary deficit 0, 10 distinct returned
- yee-pec-cube-32 / slepc: 7 of 12 pairs have backward error > tol=1e-10; completeness incomplete: 1 missing, boundary deficit 0, 12 distinct returned

## Notes

- ss-2cubes_sphere has no stored reference (a SuperLU-based one is out of reach, see eval_run2.md); both of its results are compared with the pooled accepted pairs of hsolve and slepc (uncertified).
- yee-pec-box-1e4 / slepc claims success but returns only one copy of the double eigenvalue 107.6932553 and 118.6929081 instead of the second copy.
- ss-tem27623 / slepc passes on eta (2.2e-13) yet its relative residual reaches 6.8e-5, its eigenvalue error 4.2e-7 and its largest subspace angle 5.1e-3 (hsolve: 4.1e-13, 9.3e-16, 1.0e-12). The eigenvalues (~5.9e13) exceed ||A||_2/||B||_2 (~5.0e6) by 1e7, so |lambda| ||B||_2 ||x||_2 dwarfs any residual and eta says little on this pencil.
