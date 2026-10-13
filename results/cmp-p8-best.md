# cmp-p8-best

cores `16-31` (16), SLEPc profile `best`, 5 interleaved repetition(s): medians [min-max]. Solver time = hsolve time_total_s / SLEPc -log_view setup+solve+vectors (same scope: no process start-up, loading or output). Peak GiB: guard process tree / solver-reported. Valid = exit 0, success claimed and tools/eval pass at the case tol (completeness: complete = certified by a complete reference, consistent = agrees with an incomplete one); set errors |lambda-mu|/(|mu|+||A||/||B||) of the valid runs.

| case | n | type | target | solver | configuration | solver time (s) | wall (s) | analysis | numeric (#) | solves (#/s) | ortho | iterate | verify | peak GiB | valid | evaluation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| syn-mem27-46 | 97336 | HEP | k=12 @ 0 | hsolve |  | 2.37 [2.36-2.42] | 2.48 | 0.62 | 0.48 (1) | 48 / 0.24 | - | 0.27 | 0.02 | 0.86 / 0.86 | yes (5/5) | complete; ref 2e-17; slepc 4e-16 |
| syn-mem27-46 | 97336 | HEP | k=12 @ 0 | hsolve-auto |  | 1.73 [1.72-1.73] | 1.84 | 0.62 | 0.48 (1) | 76 / 0.41 | - | 0.27 | 0.02 | 0.88 / 0.88 | yes (5/5) | complete; ref 2e-17; slepc 4e-16 |
| syn-mem27-46 | 97336 | HEP | k=12 @ 0 | hsolve-nocert |  | 1.48 [1.47-1.50] | 1.61 | 0.62 | 0.47 (1) | 48 / 0.24 | - | 0.27 | 0.02 | 0.84 / 0.84 | yes (5/5) | complete; ref 2e-17; slepc 4e-16 |
| syn-mem27-46 | 97336 | HEP | k=12 @ 0 | slepc | `1x16` (case profile syn-mem27-46) | 3.48 [3.44-3.51] | 5.95 | 1.09 | 0.52 (1) | 41 / 1.80 | 0.03 | - | - | 0.94 / 0.91 | yes (5/5) | complete; ref 4e-16 |
| yee-pec-box-1e5 | 109744 | GHEP | k=10 @ 52.85 | hsolve |  | 2.97 [2.90-3.10] | 3.08 | 0.58 | 0.52 (1) | 84 / 0.70 | - | 0.80 | 0.02 | 0.96 / 0.96 | yes (5/5) | complete; ref 3e-17; slepc 4e-14 |
| yee-pec-box-1e5 | 109744 | GHEP | k=10 @ 52.85 | hsolve-auto |  | 2.93 [2.89-3.07] | 3.02 | 0.55 | 0.51 (1) | 84 / 0.70 | - | 0.79 | 0.02 | 0.94 / 0.94 | yes (5/5) | complete; ref 3e-17; slepc 4e-14 |
| yee-pec-box-1e5 | 109744 | GHEP | k=10 @ 52.85 | hsolve-nocert |  | 2.08 [2.07-2.09] | 2.21 | 0.55 | 0.56 (1) | 84 / 0.75 | - | 0.84 | 0.02 | 0.87 / 0.87 | yes (5/5) | complete; ref 3e-17; slepc 4e-14 |
| yee-pec-box-1e5 | 109744 | GHEP | k=10 @ 52.85 | slepc | `4x4/core eps_nev=k+2 st_mat_mumps_cntl_7=1e-10 st_mat_mumps_icntl_28=2 st_mat_mumps_icntl_29=2 st_mat_mumps_icntl_35=2` (case profile yee-pec-box-1e5) | 3.35 [3.32-3.67] | 5.70 | 0.71 | 0.78 (1) | 60 / 1.75 | 0.05 | - | - | 1.43 / 1.41 | yes (5/5) | complete; ref 4e-14 |
| wg-square-1e5 | 99225 | GHIEP | k=18 @ -0.5272 | hsolve |  | 0.48 [0.48-0.49] | 0.72 | 0.05 | 0.07 (1) | 62 / 0.20 | - | 0.30 | 0.04 | 0.24 / 0.24 | yes (5/5) | consistent; slepc 2e-15 |
| wg-square-1e5 | 99225 | GHIEP | k=18 @ -0.5272 | hsolve-auto |  | 0.48 [0.48-0.54] | 0.67 | 0.05 | 0.07 (1) | 62 / 0.20 | - | 0.30 | 0.04 | 0.25 / 0.25 | yes (5/5) | consistent; slepc 2e-15 |
| wg-square-1e5 | 99225 | GHIEP | k=18 @ -0.5272 | hsolve-nocert |  | 0.49 [0.48-0.50] | 0.72 | 0.05 | 0.07 (1) | 62 / 0.20 | - | 0.30 | 0.04 | 0.24 / 0.24 | yes (5/5) | consistent; slepc 2e-15 |
| wg-square-1e5 | 99225 | GHIEP | k=18 @ -0.5272 | slepc | `16x1/core st_mat_mumps_icntl_58=1 st_mat_mumps_icntl_7=0` (case profile wg-square-1e5) | 0.92 [0.84-0.92] | 3.81 | 0.12 | 0.07 (1) | 80 / 0.56 | 0.14 | - | - | 2.20 / 2.18 | NO (4/5) | consistent; invalid: 1 of 18 pairs have backward error > tol=1e-10; solver claims no success |
| ss-qa8fk-qa8fm | 66127 | GHEP | k=10 @ 12.82 | hsolve |  | 1.70 [1.68-1.71] | 1.92 | 0.38 | 0.62 (3) | 100 / 0.42 | - | 0.52 | 0.01 | 0.61 / 0.61 | yes (5/5) | consistent; slepc 5e-15 |
| ss-qa8fk-qa8fm | 66127 | GHEP | k=10 @ 12.82 | hsolve-auto |  | 1.71 [1.70-1.89] | 1.92 | 0.38 | 0.62 (3) | 100 / 0.42 | - | 0.52 | 0.01 | 0.60 / 0.60 | yes (5/5) | consistent; slepc 5e-15 |
| ss-qa8fk-qa8fm | 66127 | GHEP | k=10 @ 12.82 | hsolve-nocert |  | 1.30 [1.29-1.36] | 1.52 | 0.38 | 0.21 (1) | 100 / 0.42 | - | 0.52 | 0.01 | 0.55 / 0.55 | yes (5/5) | consistent; slepc 5e-15 |
| ss-qa8fk-qa8fm | 66127 | GHEP | k=10 @ 12.82 | slepc | `8x2/core eps_purify=0 st_mat_mumps_icntl_28=2 st_mat_mumps_icntl_29=1` (case profile ss-qa8fk-qa8fm) | 1.49 [1.48-1.58] | 4.11 | 0.24 | 0.35 (1) | 47 / 0.80 | 0.07 | - | - | 1.74 / 1.73 | yes (5/5) | consistent |
| cav2d-rect-pert-1e5 | 99459 | GHEP | k=6 @ 0.1418 | hsolve |  | 0.30 [0.29-0.30] | 0.52 | 0.03 | 0.10 (3) | 36 / 0.07 | - | 0.11 | 0.01 | 0.24 / 0.24 | yes (5/5) | consistent; slepc 9e-19 |
| cav2d-rect-pert-1e5 | 99459 | GHEP | k=6 @ 0.1418 | hsolve-auto |  | 0.30 [0.29-0.33] | 0.52 | 0.03 | 0.10 (3) | 36 / 0.07 | - | 0.11 | 0.01 | 0.21 / 0.21 | yes (5/5) | consistent; slepc 9e-19 |
| cav2d-rect-pert-1e5 | 99459 | GHEP | k=6 @ 0.1418 | hsolve-nocert |  | 0.25 [0.23-0.28] | 0.43 | 0.03 | 0.04 (1) | 36 / 0.07 | - | 0.11 | 0.01 | 0.20 / 0.19 | yes (5/5) | consistent; slepc 9e-19 |
| cav2d-rect-pert-1e5 | 99459 | GHEP | k=6 @ 0.1418 | slepc | `16x1/core eps_ncv=2k eps_purify=0 st_mat_mumps_icntl_7=0 st_mat_mumps_icntl_8=0` (case profile cav2d-rect-pert-1e5) | 0.22 [0.22-0.22] | 3.15 | 0.09 | 0.02 (1) | 19 / 0.09 | 0.01 | - | - | 1.68 / 1.66 | yes (5/5) | consistent |

Speedup vs SLEPc (median solver time; wall in parentheses), given only when both solvers were valid in every repetition. 'vs class profile' / 'vs untuned default': SLEPc did not run a profile tuned for this case, so this is no comparison with the tuned baseline.

| case | hsolve | hsolve-auto | hsolve-nocert |
|---|---|---|---|
| syn-mem27-46 | 1.47x (2.40x) | 2.01x (3.23x) | 2.35x (3.70x) |
| yee-pec-box-1e5 | 1.13x (1.85x) | 1.14x (1.89x) | 1.61x (2.58x) |
| wg-square-1e5 | - (slepc valid 4/5) | - (slepc valid 4/5) | - (slepc valid 4/5) |
| ss-qa8fk-qa8fm | 0.87x (2.13x) | 0.87x (2.13x) | 1.15x (2.70x) |
| cav2d-rect-pert-1e5 | 0.73x (6.04x) | 0.72x (6.07x) | 0.88x (7.24x) |
