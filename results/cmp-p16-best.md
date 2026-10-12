# cmp-p16-best

cores `16-31` (16), SLEPc profile `best`, 5 interleaved repetition(s): medians [min-max]. Solver time = hsolve time_total_s / SLEPc -log_view setup+solve+vectors (same scope: no process start-up, loading or output). Peak GiB: guard process tree / solver-reported. Valid = exit 0, success claimed and tools/eval pass at the case tol (completeness: complete = certified by a complete reference, consistent = agrees with an incomplete one); set errors |lambda-mu|/(|mu|+||A||/||B||) of the valid runs.

| case | n | type | target | solver | configuration | solver time (s) | wall (s) | analysis | numeric (#) | solves (#/s) | ortho | iterate | verify | peak GiB | valid | evaluation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| syn-mem27-46 | 97336 | HEP | k=12 @ 0 | hsolve |  | 2.76 [2.76-2.80] | 2.88 | 0.64 | 0.51 (1) | 44 / 0.47 | - | 0.55 | 0.02 | 1.04 / 1.04 | yes (5/5) | complete; ref 2e-17; slepc 4e-16 |
| syn-mem27-46 | 97336 | HEP | k=12 @ 0 | hsolve-nocert |  | 1.85 [1.77-1.90] | 1.98 | 0.66 | 0.52 (1) | 44 / 0.47 | - | 0.55 | 0.02 | 1.05 / 1.05 | yes (5/5) | complete; ref 2e-17; slepc 4e-16 |
| syn-mem27-46 | 97336 | HEP | k=12 @ 0 | slepc | `1x16` (case profile syn-mem27-46) | 3.49 [3.46-3.51] | 5.89 | 1.09 | 0.52 (1) | 41 / 1.83 | 0.03 | - | - | 0.93 / 0.91 | yes (5/5) | complete; ref 4e-16 |
| yee-pec-box-1e5 | 109744 | GHEP | k=10 @ 52.85 | hsolve |  | 3.24 [3.20-3.30] | 3.38 | 0.55 | 0.51 (1) | 84 / 0.91 | - | 1.12 | 0.02 | 0.92 / 0.92 | yes (5/5) | complete; ref 3e-17; slepc 3e-14 |
| yee-pec-box-1e5 | 109744 | GHEP | k=10 @ 52.85 | hsolve-nocert |  | 2.31 [2.27-2.34] | 2.42 | 0.55 | 0.51 (1) | 84 / 0.91 | - | 1.12 | 0.02 | 0.94 / 0.94 | yes (5/5) | complete; ref 3e-17; slepc 3e-14 |
| yee-pec-box-1e5 | 109744 | GHEP | k=10 @ 52.85 | slepc | `4x4/core eps_nev=k+2 st_mat_mumps_cntl_7=1e-10 st_mat_mumps_icntl_28=2 st_mat_mumps_icntl_29=2 st_mat_mumps_icntl_35=2` (case profile yee-pec-box-1e5) | 3.35 [3.34-3.61] | 5.70 | 0.71 | 0.78 (1) | 60 / 1.76 | 0.05 | - | - | 1.43 / 1.41 | yes (5/5) | complete; ref 5e-14 |
| wg-square-1e5 | 99225 | GHIEP | k=18 @ -0.5272 | hsolve |  | 0.59 [0.59-0.61] | 0.80 | 0.05 | 0.13 (2) | 62 / 0.24 | - | 0.34 | 0.05 | 0.24 / 0.24 | yes (5/5) | consistent; slepc 2e-15 |
| wg-square-1e5 | 99225 | GHIEP | k=18 @ -0.5272 | hsolve-nocert |  | 0.59 [0.59-0.62] | 0.80 | 0.05 | 0.13 (2) | 62 / 0.25 | - | 0.35 | 0.05 | 0.27 / 0.27 | yes (5/5) | consistent; slepc 2e-15 |
| wg-square-1e5 | 99225 | GHIEP | k=18 @ -0.5272 | slepc | `16x1/core st_mat_mumps_icntl_58=1 st_mat_mumps_icntl_7=0` (case profile wg-square-1e5) | 0.89 [0.87-0.93] | 3.85 | 0.12 | 0.06 (1) | 77 / 0.54 | 0.13 | - | - | 2.20 / 2.18 | yes (5/5) | consistent |
| ss-qa8fk-qa8fm | 66127 | GHEP | k=10 @ 12.82 | hsolve |  | 2.32 [2.28-2.35] | 2.53 | 0.46 | 0.91 (4) | 100 / 0.65 | - | 0.83 | 0.01 | 0.73 / 0.73 | yes (5/5) | consistent; slepc 3e-15 |
| ss-qa8fk-qa8fm | 66127 | GHEP | k=10 @ 12.82 | hsolve-nocert |  | 1.82 [1.80-1.86] | 2.04 | 0.46 | 0.44 (2) | 100 / 0.63 | - | 0.82 | 0.01 | 0.72 / 0.72 | yes (5/5) | consistent; slepc 3e-15 |
| ss-qa8fk-qa8fm | 66127 | GHEP | k=10 @ 12.82 | slepc | `8x2/core eps_purify=0 st_mat_mumps_icntl_28=2 st_mat_mumps_icntl_29=1` (case profile ss-qa8fk-qa8fm) | 1.48 [1.45-1.49] | 4.04 | 0.24 | 0.34 (1) | 47 / 0.79 | 0.07 | - | - | 1.65 / 1.64 | yes (5/5) | consistent |
| cav2d-rect-pert-1e5 | 99459 | GHEP | k=6 @ 0.1418 | hsolve |  | 0.34 [0.34-0.34] | 0.52 | 0.03 | 0.14 (4) | 22 / 0.13 | - | 0.15 | 0.01 | 0.28 / 0.28 | yes (5/5) | consistent; slepc 9e-18 |
| cav2d-rect-pert-1e5 | 99459 | GHEP | k=6 @ 0.1418 | hsolve-nocert |  | 0.28 [0.28-0.31] | 0.47 | 0.03 | 0.08 (2) | 22 / 0.13 | - | 0.15 | 0.01 | 0.20 / 0.20 | yes (5/5) | consistent; slepc 9e-18 |
| cav2d-rect-pert-1e5 | 99459 | GHEP | k=6 @ 0.1418 | slepc | `16x1/core eps_ncv=2k eps_purify=0 st_mat_mumps_icntl_7=0 st_mat_mumps_icntl_8=0` (case profile cav2d-rect-pert-1e5) | 0.22 [0.21-0.25] | 3.15 | 0.08 | 0.02 (1) | 19 / 0.09 | 0.01 | - | - | 1.69 / 1.67 | yes (5/5) | consistent |

Speedup vs SLEPc (median solver time; wall in parentheses), given only when both solvers were valid in every repetition. 'vs class profile' / 'vs untuned default': SLEPc did not run a profile tuned for this case, so this is no comparison with the tuned baseline.

| case | hsolve | hsolve-nocert |
|---|---|---|
| syn-mem27-46 | 1.26x (2.05x) | 1.89x (2.98x) |
| yee-pec-box-1e5 | 1.03x (1.69x) | 1.45x (2.35x) |
| wg-square-1e5 | 1.50x (4.78x) | 1.51x (4.78x) |
| ss-qa8fk-qa8fm | 0.64x (1.60x) | 0.81x (1.98x) |
| cav2d-rect-pert-1e5 | 0.64x (6.09x) | 0.79x (6.74x) |
