# cmp-p5c-best

cores `16-31` (16), SLEPc profile `best`, 5 interleaved repetition(s): medians [min-max]. Solver time = hsolve time_total_s / SLEPc -log_view setup+solve+vectors (same scope: no process start-up, loading or output). Peak GiB: guard process tree / solver-reported. Valid = exit 0, success claimed and tools/eval pass at the case tol (completeness: complete = certified by a complete reference, consistent = agrees with an incomplete one); set errors |lambda-mu|/(|mu|+||A||/||B||) of the valid runs.

| case | n | type | target | solver | configuration | solver time (s) | wall (s) | analysis | numeric (#) | solves (#/s) | ortho | iterate | verify | peak GiB | valid | evaluation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| syn-mem27-46 | 97336 | HEP | k=12 @ 0 | hsolve |  | 2.39 [2.38-2.40] | 2.53 | 0.62 | 0.48 (1) | 48 / 0.24 | - | 0.27 | 0.02 | 0.85 / 0.85 | yes (5/5) | complete; ref 2e-17; slepc 4e-16 |
| syn-mem27-46 | 97336 | HEP | k=12 @ 0 | hsolve-auto |  | 1.77 [1.72-2.43] | 1.92 | 0.62 | 0.53 (1) | 76 / 0.42 | - | 0.27 | 0.02 | 0.88 / 0.88 | yes (5/5) | complete; ref 2e-17; slepc 4e-16 |
| syn-mem27-46 | 97336 | HEP | k=12 @ 0 | hsolve-nocert |  | 1.49 [1.48-1.80] | 1.61 | 0.62 | 0.48 (1) | 48 / 0.24 | - | 0.27 | 0.02 | 0.86 / 0.86 | yes (5/5) | complete; ref 2e-17; slepc 4e-16 |
| syn-mem27-46 | 97336 | HEP | k=12 @ 0 | slepc | `1x16` (case profile syn-mem27-46) | 3.48 [3.47-3.51] | 5.82 | 1.09 | 0.52 (1) | 41 / 1.81 | 0.03 | - | - | 0.92 / 0.90 | yes (5/5) | complete; ref 4e-16 |
| yee-pec-box-1e5 | 109744 | GHEP | k=10 @ 52.85 | hsolve |  | 2.95 [2.93-2.96] | 3.08 | 0.56 | 0.51 (1) | 84 / 0.72 | - | 0.81 | 0.02 | 0.95 / 0.95 | yes (5/5) | complete; ref 3e-17; slepc 2e-14 |
| yee-pec-box-1e5 | 109744 | GHEP | k=10 @ 52.85 | hsolve-auto |  | 2.94 [2.91-3.03] | 3.08 | 0.58 | 0.51 (1) | 84 / 0.70 | - | 0.79 | 0.02 | 0.93 / 0.93 | yes (5/5) | complete; ref 3e-17; slepc 2e-14 |
| yee-pec-box-1e5 | 109744 | GHEP | k=10 @ 52.85 | hsolve-nocert |  | 2.10 [2.09-2.15] | 2.22 | 0.54 | 0.58 (1) | 84 / 0.75 | - | 0.84 | 0.02 | 0.87 / 0.87 | yes (5/5) | complete; ref 3e-17; slepc 2e-14 |
| yee-pec-box-1e5 | 109744 | GHEP | k=10 @ 52.85 | slepc | `4x4/core eps_nev=k+2 st_mat_mumps_cntl_7=1e-10 st_mat_mumps_icntl_28=2 st_mat_mumps_icntl_29=2 st_mat_mumps_icntl_35=2` (case profile yee-pec-box-1e5) | 3.36 [3.33-3.37] | 5.75 | 0.71 | 0.79 (1) | 60 / 1.76 | 0.04 | - | - | 1.37 / 1.35 | yes (5/5) | complete; ref 5e-14 |
| wg-square-1e5 | 99225 | GHIEP | k=18 @ -0.5272 | hsolve |  | 0.48 [0.48-0.55] | 0.72 | 0.05 | 0.07 (1) | 62 / 0.20 | - | 0.30 | 0.04 | 0.24 / 0.24 | yes (5/5) | consistent; slepc 2e-15 |
| wg-square-1e5 | 99225 | GHIEP | k=18 @ -0.5272 | hsolve-auto |  | 0.49 [0.48-0.49] | 0.72 | 0.05 | 0.07 (1) | 62 / 0.20 | - | 0.30 | 0.04 | 0.24 / 0.24 | yes (5/5) | consistent; slepc 2e-15 |
| wg-square-1e5 | 99225 | GHIEP | k=18 @ -0.5272 | hsolve-nocert |  | 0.49 [0.48-0.70] | 0.72 | 0.05 | 0.07 (1) | 62 / 0.20 | - | 0.30 | 0.04 | 0.22 / 0.22 | yes (5/5) | consistent; slepc 2e-15 |
| wg-square-1e5 | 99225 | GHIEP | k=18 @ -0.5272 | slepc | `16x1/core st_mat_mumps_icntl_58=1 st_mat_mumps_icntl_7=0` (case profile wg-square-1e5) | 0.89 [0.86-0.92] | 3.44 | 0.12 | 0.07 (1) | 76 / 0.54 | 0.13 | - | - | 1.94 / 1.92 | yes (5/5) | consistent |
| ss-qa8fk-qa8fm | 66127 | GHEP | k=10 @ 12.82 | hsolve |  | 1.68 [1.67-1.69] | 1.92 | 0.37 | 0.61 (3) | 100 / 0.42 | - | 0.52 | 0.01 | 0.61 / 0.61 | yes (5/5) | consistent; slepc 2e-15 |
| ss-qa8fk-qa8fm | 66127 | GHEP | k=10 @ 12.82 | hsolve-auto |  | 1.69 [1.68-1.69] | 1.88 | 0.37 | 0.61 (3) | 100 / 0.42 | - | 0.52 | 0.01 | 0.61 / 0.62 | yes (5/5) | consistent; slepc 2e-15 |
| ss-qa8fk-qa8fm | 66127 | GHEP | k=10 @ 12.82 | hsolve-nocert |  | 1.27 [1.27-1.28] | 1.48 | 0.37 | 0.21 (1) | 100 / 0.42 | - | 0.51 | 0.01 | 0.53 / 0.53 | yes (5/5) | consistent; slepc 2e-15 |
| ss-qa8fk-qa8fm | 66127 | GHEP | k=10 @ 12.82 | slepc | `8x2/core eps_purify=0 st_mat_mumps_icntl_28=2 st_mat_mumps_icntl_29=1` (case profile ss-qa8fk-qa8fm) | 1.50 [1.48-1.54] | 4.06 | 0.23 | 0.35 (1) | 47 / 0.81 | 0.07 | - | - | 1.55 / 1.54 | yes (5/5) | consistent |
| cav2d-rect-pert-1e5 | 99459 | GHEP | k=6 @ 0.1418 | hsolve |  | 0.30 [0.29-0.34] | 0.52 | 0.03 | 0.10 (3) | 36 / 0.07 | - | 0.11 | 0.01 | 0.24 / 0.24 | yes (5/5) | consistent; slepc 5e-18 |
| cav2d-rect-pert-1e5 | 99459 | GHEP | k=6 @ 0.1418 | hsolve-auto |  | 0.30 [0.29-0.32] | 0.52 | 0.03 | 0.10 (3) | 36 / 0.07 | - | 0.11 | 0.01 | 0.25 / 0.25 | yes (5/5) | consistent; slepc 5e-18 |
| cav2d-rect-pert-1e5 | 99459 | GHEP | k=6 @ 0.1418 | hsolve-nocert |  | 0.23 [0.23-0.27] | 0.41 | 0.03 | 0.04 (1) | 36 / 0.07 | - | 0.11 | 0.01 | 0.21 / 0.21 | yes (5/5) | consistent; slepc 5e-18 |
| cav2d-rect-pert-1e5 | 99459 | GHEP | k=6 @ 0.1418 | slepc | `16x1/core eps_ncv=2k eps_purify=0 st_mat_mumps_icntl_7=0 st_mat_mumps_icntl_8=0` (case profile cav2d-rect-pert-1e5) | 0.21 [0.21-0.22] | 2.80 | 0.08 | 0.02 (1) | 19 / 0.09 | 0.01 | - | - | 1.42 / 1.40 | yes (5/5) | consistent |

Speedup vs SLEPc (median solver time; wall in parentheses), given only when both solvers were valid in every repetition. 'vs class profile' / 'vs untuned default': SLEPc did not run a profile tuned for this case, so this is no comparison with the tuned baseline.

| case | hsolve | hsolve-auto | hsolve-nocert |
|---|---|---|---|
| syn-mem27-46 | 1.45x (2.30x) | 1.96x (3.03x) | 2.34x (3.62x) |
| yee-pec-box-1e5 | 1.14x (1.86x) | 1.14x (1.86x) | 1.60x (2.59x) |
| wg-square-1e5 | 1.84x (4.79x) | 1.82x (4.79x) | 1.83x (4.78x) |
| ss-qa8fk-qa8fm | 0.89x (2.11x) | 0.89x (2.16x) | 1.18x (2.75x) |
| cav2d-rect-pert-1e5 | 0.71x (5.38x) | 0.71x (5.39x) | 0.94x (6.82x) |

Parts (one compare.py call each; all solvers of a case ran on the same node): cmp-p5c-best-c003 on c003: syn-mem27-46, yee-pec-box-1e5, wg-square-1e5; cmp-p5c-best-c005 on c005: ss-qa8fk-qa8fm, cav2d-rect-pert-1e5.
