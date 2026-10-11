# Evaluation of run2 (hsolve, all cases)

Run `/home/zhlin/gencode/os2026/results/raw/run2`, data root `/home/zhlin/gencode/os2026/data`; 39 evaluation(s) by `python -m tools.eval.sweep` (tools/eval/sweep.py), evaluation JSONs in `results/raw/eval/run2/<case>/<solver>/evaluation.json`.

Verdicts - **hsolve**: 39 pass / 0 fail

Metrics (tools/eval/evaluate.py, recomputed from the case matrices and the returned vectors): ref = reference kind used (discrete-exact / analytic / dense: certified complete; solver: ARPACK shift-invert, uncertified; pooled: accepted pairs of all solvers of this run, uncertified; n/c = not comparable, e.g. continuum-model values; - = none); time = solver-reported `time_total_s`; wall / peak = wall time and peak RSS of the whole process tree measured by tools/run/guard.py (peak marked * is solver-reported, when no guard summary exists); eta = backward error ||Ax - lambda Bx||_2 / ((||A||_2 + |lambda| ||B||_2) ||x||_2); rel res = ||Ax - lambda Bx||_2 / (|lambda| ||Bx||_2), which stays informative where |lambda| ||B||_2 ||x||_2 dwarfs the residual and eta tells little (e.g. a badly conditioned B); eig err = |lambda - mu|/|mu| / |lambda - mu|/(|mu| + ||A||_2/||B||_2) against the matched reference value (not against a pooled reference, which holds the solvers' own values); angle = largest principal angle between computed and reference eigenspaces per eigenvalue cluster (needs reference eigenvectors); B-orth = ||X^H B X - I||_2 (HEP/GHEP; not part of the verdict); completeness = status (found/expected, + pairs outside the target set): complete/incomplete against a certified reference, consistent/inconsistent against an uncertified one; claim = the solver's own success flag vs the verdict (over = success claimed but the evaluation fails, under = failure reported but the evaluation passes). Verdict pass = every eta <= tol, no duplicate vectors, completeness not incomplete/inconsistent, no spurious eigenvalue.

| case | solver | problem | n | k | ref | time [s] | wall [s] | peak [MiB] | max eta | max rel res | max eig err (rel / scaled) | max angle | B-orth | completeness | claim | verdict |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| cav2d-rect-pert-1e5 | hsolve | GHEP | 99459 | 6 | solver | 1.74 | 2.01 | 115 | 3.4e-15 | 3.4e-10 | 1.2e-13 / 1.9e-18 | 1.3e-11 | 1.8e-15 | consistent (6/6) | ok | pass |
| cav2d-square-1e4 | hsolve | GHEP | 9976 | 7 | solver | 0.101 | 0.203 | 73 | 1.4e-14 | 4.7e-11 | 1.4e-13 / 3.4e-17 | 5.9e-13 | 6.6e-15 | consistent (7/7) | ok | pass |
| ss-2cubes_sphere | hsolve | HEP | 101492 | 10 | - | 8.53 | 8.82 | 570 | 9.9e-15 | 9.9e-13 | - / - | - | 1.5e-15 | unknown | ok | pass |
| ss-ge87h76 | hsolve | HEP | 112985 | 10 | - | 265 | 266 | 3851 | 6.3e-13 | 1.1e-08 | - / - | - | 2.0e-15 | unknown | ok | pass |
| ss-helm3d01 | hsolve | HEP | 32226 | 10 | solver | 1.34 | 1.61 | 116 | 4.4e-13 | 3.9e-08 | 7.9e-13 / 1.8e-17 | 7.7e-12 | 2.0e-15 | consistent (10/10) | ok | pass |
| ss-qa8fk-qa8fm | hsolve | GHEP | 66127 | 10 | solver | 5.85 | 6.22 | 362 | 1.7e-13 | 7.2e-12 | 8.3e-15 / 2.4e-16 | 6.3e-12 | 1.9e-15 | consistent (10/10) | ok | pass |
| ss-si34h36 | hsolve | HEP | 97569 | 10 | - | 159 | 159 | 3088 | 4.7e-13 | 4.5e-09 | - / - | - | 1.8e-15 | unknown | ok | pass |
| ss-tem27623 | hsolve | GHEP | 27623 | 10 | solver | 1.35 | 1.81 | 109 | 1.1e-21 | 3.0e-13 | 8.0e-16 / 8.0e-16 | 6.7e-13 | 2.0e-15 | consistent (10/10) | ok | pass |
| syn-ghep-multiplicity-real-1e4 | hsolve | GHEP | 10000 | 22 | discrete-exact | 0.256 | 0.404 | 73 | 1.4e-16 | 1.3e-14 | 2.0e-15 / 9.0e-16 | - | 2.0e-15 | complete (22/22) | ok | pass |
| syn-ghep-near-singular-real-1e5 | hsolve | GHEP | 100000 | 10 | discrete-exact | 2.49 | 3.01 | 197 | 1.7e-15 | 4.8e-14 | 9.3e-16 / 2.8e-16 | - | 1.9e-15 | complete (10/10) | ok | pass |
| syn-ghep-zero-cluster-complex-1e4 | hsolve | GHEP | 10000 | 26 | discrete-exact | 0.784 | 1 | 90 | 3.0e-13 | 6.2e-07 | 1.4e-11 / 1.3e-16 | - | 3.3e-15 | complete (26/26) | ok | pass |
| syn-lap2d-100 | hsolve | HEP | 10000 | 12 | discrete-exact | 0.148 | 0.203 | 73 | 1.2e-14 | 7.1e-11 | 4.0e-13 / 3.2e-17 | - | 2.7e-15 | complete (12/12) | ok | pass |
| syn-lap3d-22 | hsolve | HEP | 10648 | 12 | discrete-exact | 0.382 | 0.604 | 73 | 8.5e-14 | 9.0e-10 | 6.5e-13 / 5.5e-17 | - | 1.7e-15 | complete (12/12) | ok | pass |
| syn-lap3d-aniso-1e5 | hsolve | HEP | 97200 | 12 | discrete-exact | 5.81 | 6.02 | 440 | 1.2e-12 | 7.4e-07 | 2.0e-11 / 5.1e-17 | - | 2.1e-15 | complete (12/12) | ok | pass |
| syn-mem125-46 | hsolve | HEP | 97336 | 15 | discrete-exact | 34.2 | 34.7 | 2028 | 2.3e-12 | 7.5e-09 | 2.5e-13 / 7.4e-17 | - | 2.8e-15 | complete (15/15) | ok | pass |
| syn-mem27-46 | hsolve | HEP | 97336 | 12 | discrete-exact | 9.39 | 9.63 | 649 | 6.1e-13 | 6.4e-07 | 1.4e-11 / 1.7e-17 | - | 1.8e-15 | complete (12/12) | ok | pass |
| syn-multiplicity-complex-1e4 | hsolve | HEP | 10000 | 14 | discrete-exact | 0.621 | 0.805 | 73 | 6.7e-14 | 2.2e-13 | 1.0e-15 / 3.1e-16 | - | 1.9e-15 | complete (14/14) | ok | pass |
| syn-multiplicity-real-1e4 | hsolve | HEP | 10000 | 14 | discrete-exact | 0.27 | 0.403 | 73 | 4.1e-14 | 1.4e-13 | 1.2e-15 / 3.5e-16 | - | 1.6e-15 | complete (14/14) | ok | pass |
| syn-near-singular-complex-1e4 | hsolve | HEP | 10000 | 10 | discrete-exact | 0.502 | 0.604 | 73 | 9.2e-12 | 4.0e-11 | 1.7e-15 / 3.8e-16 | - | 2.0e-15 | complete (10/10) | ok | pass |
| syn-near-singular-complex-1e5 | hsolve | HEP | 100000 | 10 | discrete-exact | 8.57 | 9.03 | 440 | 4.2e-11 | 1.8e-10 | 9.3e-16 / 2.1e-16 | - | 1.5e-15 | complete (10/10) | ok | pass |
| syn-near-singular-real-1e4 | hsolve | HEP | 10000 | 10 | discrete-exact | 0.138 | 0.404 | 73 | 2.5e-14 | 1.1e-13 | 9.3e-16 / 2.1e-16 | - | 2.8e-15 | complete (10/10) | ok | pass |
| syn-wide-range-complex-1e4 | hsolve | HEP | 10000 | 10 | discrete-exact | 0.427 | 0.604 | 73 | 1.8e-16 | 1.8e-10 | 8.8e-16 / 8.9e-22 | - | 2.1e-15 | complete (10/10) | ok | pass |
| syn-wide-range-real-1e4 | hsolve | HEP | 10000 | 10 | discrete-exact | 0.0964 | 0.203 | 73 | 1.2e-16 | 1.2e-10 | 1.5e-15 / 1.6e-21 | - | 1.3e-15 | complete (10/10) | ok | pass |
| syn-zero-cluster-complex-1e4 | hsolve | HEP | 10000 | 20 | discrete-exact | 3.98 | 4.21 | 78 | 1.1e-15 | 4.3e-06 | 7.0e-07 / 2.4e-17 | - | 3.1e-15 | complete (20/20) | ok | pass |
| syn-zero-cluster-real-1e4 | hsolve | HEP | 10000 | 20 | discrete-exact | 0.893 | 1.01 | 73 | 2.2e-15 | 1.3e-05 | 5.5e-07 / 4.9e-17 | - | 4.5e-15 | complete (20/20) | ok | pass |
| syn-zero-cluster-real-1e5 | hsolve | HEP | 100000 | 21 | discrete-exact | 5.4 | 5.82 | 348 | 3.6e-15 | 3.5e-06 | 1.7e-08 / 6.9e-17 | - | 3.5e-15 | complete (21/21) | ok | pass |
| wg-halfslab-1e4 | hsolve | GHIEP | 9801 | 23 | solver | 0.242 | 0.604 | 73 | 1.5e-13 | 1.9e-10 | 5.9e-14 / 5.9e-16 | 2.2e-10 | - | consistent (23/23) | ok | pass |
| wg-halfslab-1e5 | hsolve | GHIEP | 99225 | 28 | solver | 5.11 | 5.42 | 409 | 1.4e-15 | 7.0e-09 | 3.2e-10 / 9.3e-15 | 2.0e-10 | - | consistent (28/28) | ok | pass |
| wg-square-1e3 | hsolve | GHIEP | 961 | 18 | dense | 0.0216 | 0.203 | 73 | 1.0e-14 | 3.8e-12 | 7.4e-13 / 1.7e-14 | 4.6e-12 | - | complete (18/18) | ok | pass |
| wg-square-1e5 | hsolve | GHIEP | 99225 | 18 | solver | 3.46 | 3.61 | 300 | 7.0e-16 | 3.0e-10 | 6.3e-10 / 2.1e-13 | 5.9e-08 | - | consistent (18/18) | ok | pass |
| wg-square-n289 | hsolve | GHIEP | 289 | 18 | dense | 0.028 | 0.203 | 73 | 1.5e-14 | 1.9e-11 | 1.1e-11 / 5.6e-14 | 1.4e-12 | - | complete (18/18) | ok | pass |
| wg-square-pert-1e4 | hsolve | GHIEP | 9801 | 18 | solver | 0.188 | 0.604 | 73 | 7.0e-16 | 1.8e-11 | 3.3e-13 / 8.2e-16 | 1.1e-11 | - | consistent (18/18) | ok | pass |
| yee-bloch-1e4 | hsolve | HEP | 10080 | 10 | discrete-exact | 2.05 | 2.21 | 209 | 1.1e-13 | 1.2e-11 | 4.7e-15 / 7.7e-17 | - | 1.7e-15 | complete (10/10) | ok | pass |
| yee-pec-box-1e4 | hsolve | GHEP | 10064 | 11 | discrete-exact | 0.396 | 0.604 | 73 | 3.5e-14 | 2.7e-12 | 7.9e-16 / 1.5e-17 | - | 2.0e-15 | complete (11/11) | ok | pass |
| yee-pec-box-1e5 | hsolve | GHEP | 109744 | 10 | discrete-exact | 13.7 | 13.8 | 632 | 4.1e-13 | 3.6e-10 | 2.2e-14 / 3.0e-17 | - | 1.5e-15 | complete (10/10) | ok | pass |
| yee-pec-cube-16 | hsolve | HEP | 10800 | 12 | discrete-exact | 0.466 | 0.604 | 73 | 2.0e-14 | 1.1e-12 | 1.2e-15 / 1.8e-17 | - | 1.6e-15 | complete (12/12) | ok | pass |
| yee-pec-cube-32 | hsolve | HEP | 92256 | 12 | discrete-exact | 8.5 | 8.63 | 562 | 7.4e-14 | 1.5e-11 | 1.9e-15 / 9.2e-18 | - | 1.9e-15 | complete (12/12) | ok | pass |
| yee-photonic-rods-1e4 | hsolve | HEP | 10125 | 10 | solver | 2.1 | 2.21 | 213 | 1.9e-13 | 1.7e-11 | 2.3e-15 / 2.6e-17 | 1.8e-12 | 2.8e-15 | consistent (10/10) | ok | pass |
| yee-photonic-spheres-1e4 | hsolve | GHEP | 10125 | 10 | solver | 2.22 | 2.41 | 213 | 1.3e-12 | 2.3e-11 | 1.6e-15 / 2.1e-16 | 7.2e-12 | 1.8e-15 | consistent (10/10) | ok | pass |

## Notes

- ss-2cubes_sphere, ss-ge87h76, ss-si34h36 have no reference, so their completeness is unknown (residuals, duplicates and B-orthogonality are still checked). A tools/eval/reference.py solver reference needs a SuperLU factorization of A - sigma I: SuperLU had not finished factorizing 2cubes_sphere (MMD_AT_PLUS_A ordering) after 600 s at 4.7 GiB peak RSS, and Si34H36 / Ge87H76 have 3.1x / 4.8x its nonzeros.
- The four zero-cluster cases show relative residuals up to 1.3e-5 because |lambda| is as small as 2.4e-11 in the denominator; their backward errors (<= 3.0e-13) and scaled eigenvalue errors (<= 1.3e-16) are the meaningful measures.
