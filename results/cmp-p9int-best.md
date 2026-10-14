# cmp-p9int-best

cores `16-31` (16), SLEPc profile `best`, 5 interleaved repetition(s): medians [min-max]. Solver time = hsolve time_total_s / SLEPc -log_view setup+solve+vectors (same scope: no process start-up, loading or output). Peak GiB: guard process tree / solver-reported. Valid = exit 0, success claimed and tools/eval pass at the case tol (completeness: complete = certified by a complete reference, consistent = agrees with an incomplete one); set errors |lambda-mu|/(|mu|+||A||/||B||) of the valid runs.

| case | n | type | target | solver | configuration | solver time (s) | wall (s) | analysis | numeric (#) | solves (#/s) | ortho | iterate | verify | peak GiB | valid | evaluation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| syn-mem27-46 | 97336 | HEP | k=12 @ 0 | hsolve |  | 1.99 [1.98-2.05] | 2.13 | 0.29 | 0.46 (1) | 48 / 0.23 | - | 0.26 | 0.02 | 0.84 / 0.84 | yes (5/5) | complete; ref 2e-17; slepc 4e-16 |
| syn-mem27-46 | 97336 | HEP | k=12 @ 0 | hsolve-auto |  | 1.36 [1.36-1.51] | 1.52 | 0.29 | 0.46 (1) | 76 / 0.40 | - | 0.26 | 0.02 | 0.86 / 0.86 | yes (5/5) | complete; ref 2e-17; slepc 4e-16 |
| syn-mem27-46 | 97336 | HEP | k=12 @ 0 | hsolve-nocert |  | 1.12 [1.12-1.12] | 1.24 | 0.26 | 0.47 (1) | 48 / 0.24 | - | 0.27 | 0.02 | 0.81 / 0.80 | yes (5/5) | complete; ref 2e-17; slepc 4e-16 |
| syn-mem27-46 | 97336 | HEP | k=12 @ 0 | slepc | `1x16` (case profile syn-mem27-46) | 3.51 [3.49-3.52] | 5.82 | 1.09 | 0.53 (1) | 41 / 1.83 | 0.03 | - | - | 0.93 / 0.92 | yes (5/5) | complete; ref 4e-16 |
| yee-pec-box-1e5 | 109744 | GHEP | k=10 @ 52.85 | hsolve |  | 2.44 [2.42-2.47] | 2.58 | 0.26 | 0.45 (1) | 84 / 0.67 | - | 0.75 | 0.02 | 0.77 / 0.77 | yes (5/5) | complete; ref 3e-17; slepc 2e-14 |
| yee-pec-box-1e5 | 109744 | GHEP | k=10 @ 52.85 | hsolve-auto |  | 2.44 [2.43-2.46] | 2.58 | 0.26 | 0.45 (1) | 84 / 0.66 | - | 0.75 | 0.02 | 0.76 / 0.76 | yes (5/5) | complete; ref 3e-17; slepc 2e-14 |
| yee-pec-box-1e5 | 109744 | GHEP | k=10 @ 52.85 | hsolve-nocert |  | 1.62 [1.60-1.62] | 1.72 | 0.26 | 0.45 (1) | 84 / 0.68 | - | 0.77 | 0.02 | 0.74 / 0.74 | yes (5/5) | complete; ref 3e-17; slepc 2e-14 |
| yee-pec-box-1e5 | 109744 | GHEP | k=10 @ 52.85 | slepc | `4x4/core eps_nev=k+2 st_mat_mumps_cntl_7=1e-10 st_mat_mumps_icntl_28=2 st_mat_mumps_icntl_29=2 st_mat_mumps_icntl_35=2` (case profile yee-pec-box-1e5) | 3.34 [3.33-3.36] | 5.75 | 0.72 | 0.78 (1) | 60 / 1.75 | 0.05 | - | - | 1.43 / 1.40 | yes (5/5) | complete; ref 3e-14 |
| wg-square-1e5 | 99225 | GHIEP | k=18 @ -0.5272 | hsolve |  | 0.48 [0.48-0.52] | 0.72 | 0.05 | 0.07 (1) | 62 / 0.20 | - | 0.30 | 0.04 | 0.30 / 0.30 | yes (5/5) | consistent; slepc 1e-14 |
| wg-square-1e5 | 99225 | GHIEP | k=18 @ -0.5272 | hsolve-auto |  | 0.48 [0.48-0.48] | 0.67 | 0.05 | 0.06 (1) | 62 / 0.20 | - | 0.30 | 0.04 | 0.30 / 0.30 | yes (5/5) | consistent; slepc 1e-14 |
| wg-square-1e5 | 99225 | GHIEP | k=18 @ -0.5272 | hsolve-nocert |  | 0.48 [0.47-0.50] | 0.67 | 0.05 | 0.06 (1) | 62 / 0.20 | - | 0.30 | 0.04 | 0.31 / 0.31 | yes (5/5) | consistent; slepc 1e-14 |
| wg-square-1e5 | 99225 | GHIEP | k=18 @ -0.5272 | slepc | `16x1/core st_mat_mumps_icntl_58=1 st_mat_mumps_icntl_7=0` (case profile wg-square-1e5) | 0.91 [0.84-0.93] | 3.83 | 0.12 | 0.07 (1) | 80 / 0.55 | 0.14 | - | - | 2.20 / 2.18 | yes (5/5) | consistent |
| ss-qa8fk-qa8fm | 66127 | GHEP | k=10 @ 12.82 | hsolve |  | 1.49 [1.49-1.50] | 1.72 | 0.20 | 0.58 (3) | 100 / 0.42 | - | 0.52 | 0.01 | 0.74 / 0.74 | yes (5/5) | consistent; slepc 7e-16 |
| ss-qa8fk-qa8fm | 66127 | GHEP | k=10 @ 12.82 | hsolve-auto |  | 1.50 [1.48-1.51] | 1.72 | 0.20 | 0.58 (3) | 100 / 0.42 | - | 0.52 | 0.01 | 0.73 / 0.73 | yes (5/5) | consistent; slepc 7e-16 |
| ss-qa8fk-qa8fm | 66127 | GHEP | k=10 @ 12.82 | hsolve-nocert |  | 1.10 [1.09-1.10] | 1.32 | 0.19 | 0.20 (1) | 100 / 0.42 | - | 0.52 | 0.01 | 0.48 / 0.48 | yes (5/5) | consistent; slepc 7e-16 |
| ss-qa8fk-qa8fm | 66127 | GHEP | k=10 @ 12.82 | slepc | `8x2/core eps_purify=0 st_mat_mumps_icntl_28=2 st_mat_mumps_icntl_29=1` (case profile ss-qa8fk-qa8fm) | 1.43 [1.42-1.50] | 4.03 | 0.23 | 0.33 (1) | 47 / 0.78 | 0.07 | - | - | 1.65 / 1.64 | yes (5/5) | consistent |
| cav2d-rect-pert-1e5 | 99459 | GHEP | k=6 @ 0.1418 | hsolve |  | 0.29 [0.29-0.30] | 0.52 | 0.03 | 0.10 (3) | 36 / 0.07 | - | 0.11 | 0.01 | 0.25 / 0.25 | yes (5/5) | consistent; slepc 6e-18 |
| cav2d-rect-pert-1e5 | 99459 | GHEP | k=6 @ 0.1418 | hsolve-auto |  | 0.30 [0.29-0.31] | 0.52 | 0.03 | 0.10 (3) | 36 / 0.07 | - | 0.11 | 0.01 | 0.25 / 0.25 | yes (5/5) | consistent; slepc 6e-18 |
| cav2d-rect-pert-1e5 | 99459 | GHEP | k=6 @ 0.1418 | hsolve-nocert |  | 0.23 [0.23-0.25] | 0.42 | 0.03 | 0.04 (1) | 36 / 0.07 | - | 0.11 | 0.01 | 0.24 / 0.24 | yes (5/5) | consistent; slepc 6e-18 |
| cav2d-rect-pert-1e5 | 99459 | GHEP | k=6 @ 0.1418 | slepc | `16x1/core eps_ncv=2k eps_purify=0 st_mat_mumps_icntl_7=0 st_mat_mumps_icntl_8=0` (case profile cav2d-rect-pert-1e5) | 0.22 [0.22-0.22] | 3.15 | 0.08 | 0.02 (1) | 19 / 0.09 | 0.01 | - | - | 1.68 / 1.66 | yes (5/5) | consistent |

Speedup vs SLEPc (median solver time; wall in parentheses), given only when both solvers were valid in every repetition. 'vs class profile' / 'vs untuned default': SLEPc did not run a profile tuned for this case, so this is no comparison with the tuned baseline.

| case | hsolve | hsolve-auto | hsolve-nocert |
|---|---|---|---|
| syn-mem27-46 | 1.77x (2.73x) | 2.57x (3.83x) | 3.15x (4.71x) |
| yee-pec-box-1e5 | 1.37x (2.23x) | 1.37x (2.23x) | 2.07x (3.34x) |
| wg-square-1e5 | 1.89x (5.33x) | 1.90x (5.73x) | 1.89x (5.72x) |
| ss-qa8fk-qa8fm | 0.96x (2.34x) | 0.96x (2.34x) | 1.31x (3.05x) |
| cav2d-rect-pert-1e5 | 0.74x (6.04x) | 0.74x (6.10x) | 0.94x (7.53x) |

Parts (one compare.py call each; all solvers of a case ran on the same node): cmp-p9int-best on gpu01: syn-mem27-46, yee-pec-box-1e5, wg-square-1e5, ss-qa8fk-qa8fm, cav2d-rect-pert-1e5.
