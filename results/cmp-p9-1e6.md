# cmp-p9-1e6

cores `16-31` (16), SLEPc profile `best`, 5 interleaved repetition(s): medians [min-max]. Solver time = hsolve time_total_s / SLEPc -log_view setup+solve+vectors (same scope: no process start-up, loading or output). Peak GiB: guard process tree / solver-reported. Valid = exit 0, success claimed and tools/eval pass at the case tol (completeness: complete = certified by a complete reference, consistent = agrees with an incomplete one); set errors |lambda-mu|/(|mu|+||A||/||B||) of the valid runs.

| case | n | type | target | solver | configuration | solver time (s) | wall (s) | analysis | numeric (#) | solves (#/s) | ortho | iterate | verify | peak GiB | valid | evaluation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| syn-mem27-100 | 1000000 | HEP | k=12 @ 0 | hsolve |  | 111.61 [102.62-117.07] | 112.38 | 6.21 | 29.67 (1) | 168 / 15.84 | - | 17.44 | 0.26 | 16.02 / 16.02 | yes (5/5) | complete; ref 1e-16; slepc 1e-15 |
| syn-mem27-100 | 1000000 | HEP | k=12 @ 0 | hsolve-auto |  | 72.44 [67.16-74.35] | 73.33 | 6.16 | 29.52 (1) | 252 / 31.36 | - | 17.47 | 0.26 | 16.64 / 16.65 | yes (5/5) | complete; ref 1e-16; slepc 1e-15 |
| syn-mem27-100 | 1000000 | HEP | k=12 @ 0 | hsolve-nocert |  | 54.41 [49.88-57.06] | 55.27 | 6.08 | 28.92 (1) | 168 / 16.27 | - | 17.86 | 0.26 | 15.56 / 15.56 | yes (5/5) | complete; ref 1e-16; slepc 1e-15 |
| syn-mem27-100 | 1000000 | HEP | k=12 @ 0 | slepc | `1x16 eps_ncv=3k` (case profile syn-mem27-100) | 115.68 [115.49-115.83] | 119.70 | 14.20 | 23.22 (1) | 105 / 75.94 | 1.98 | - | - | 12.68 / 12.49 | yes (5/5) | complete; ref 1e-15 |
| yee-pec-box-1e6 | 1027733 | GHEP | k=10 @ 52.9 | hsolve |  | 88.91 [87.89-89.61] | 89.58 | 4.45 | 23.46 (1) | 136 / 11.97 | - | 14.23 | 0.22 | 13.00 / 13.00 | yes (5/5) | complete; ref 3e-17; slepc 1e-14 |
| yee-pec-box-1e6 | 1027733 | GHEP | k=10 @ 52.9 | hsolve-auto |  | 65.53 [63.69-67.07] | 66.39 | 4.51 | 23.54 (1) | 258 / 30.58 | - | 14.39 | 0.22 | 14.23 / 14.23 | yes (5/5) | complete; ref 3e-17; slepc 1e-14 |
| yee-pec-box-1e6 | 1027733 | GHEP | k=10 @ 52.9 | hsolve-nocert |  | 43.12 [42.77-43.92] | 43.92 | 4.60 | 22.95 (1) | 136 / 11.92 | - | 14.19 | 0.22 | 12.82 / 12.82 | yes (5/5) | complete; ref 3e-17; slepc 1e-14 |
| yee-pec-box-1e6 | 1027733 | GHEP | k=10 @ 52.9 | slepc | `4x4/core eps_ncv=3k eps_nev=k+2 st_mat_mumps_cntl_7=1e-10 st_mat_mumps_icntl_28=2 st_mat_mumps_icntl_29=2 st_mat_mumps_icntl_35=2` (case profile yee-pec-box-1e6) | 59.58 [57.97-59.80] | 63.05 | 6.73 | 24.39 (1) | 60 / 27.25 | 0.75 | - | - | 15.40 / 15.24 | yes (5/5) | complete; ref 4e-14 |

Speedup vs SLEPc (median solver time; wall in parentheses), given only when both solvers were valid in every repetition. 'vs class profile' / 'vs untuned default': SLEPc did not run a profile tuned for this case, so this is no comparison with the tuned baseline.

| case | hsolve | hsolve-auto | hsolve-nocert |
|---|---|---|---|
| syn-mem27-100 | 1.04x (1.07x) | 1.60x (1.63x) | 2.13x (2.17x) |
| yee-pec-box-1e6 | 0.67x (0.70x) | 0.91x (0.95x) | 1.38x (1.44x) |

Parts (one compare.py call each; all solvers of a case ran on the same node): cmp-p9fix-1e6-syn-mem27-100 on c005: syn-mem27-100; cmp-p9fix-1e6-yee-pec-box-1e6 on c005: yee-pec-box-1e6.
