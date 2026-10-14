# cmp-p9-1e6

cores `16-31` (16), SLEPc profile `best`, 3 interleaved repetition(s): medians [min-max]. Solver time = hsolve time_total_s / SLEPc -log_view setup+solve+vectors (same scope: no process start-up, loading or output). Peak GiB: guard process tree / solver-reported. Valid = exit 0, success claimed and tools/eval pass at the case tol (completeness: complete = certified by a complete reference, consistent = agrees with an incomplete one); set errors |lambda-mu|/(|mu|+||A||/||B||) of the valid runs.

| case | n | type | target | solver | configuration | solver time (s) | wall (s) | analysis | numeric (#) | solves (#/s) | ortho | iterate | verify | peak GiB | valid | evaluation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| syn-mem27-100 | 1000000 | HEP | k=12 @ 0 | hsolve |  | 108.02 [103.47-108.81] | 108.86 | 6.08 | 28.49 (1) | 168 / 15.72 | - | 17.31 | 0.26 | 15.21 / 15.21 | yes (3/3) | complete; ref 1e-16; slepc 7e-16 |
| syn-mem27-100 | 1000000 | HEP | k=12 @ 0 | hsolve-auto |  | 71.79 [69.27-72.48] | 72.52 | 5.86 | 29.41 (1) | 260 / 31.09 | - | 17.79 | 0.26 | 16.25 / 16.25 | yes (3/3) | complete; ref 1e-16; slepc 7e-16 |
| syn-mem27-100 | 1000000 | HEP | k=12 @ 0 | hsolve-nocert |  | 53.18 [52.19-56.43] | 54.03 | 5.88 | 28.14 (1) | 168 / 16.05 | - | 17.66 | 0.26 | 15.50 / 15.50 | yes (3/3) | complete; ref 1e-16; slepc 7e-16 |
| syn-mem27-100 | 1000000 | HEP | k=12 @ 0 | slepc | `1x16` (case profile syn-mem27-100) | 123.90 [117.27-124.95] | 127.99 | 14.29 | 23.23 (1) | 117 / 84.26 | 1.75 | - | - | 12.62 / 12.43 | yes (3/3) | complete; ref 1e-15 |
| yee-pec-box-1e6 | 1027733 | GHEP | k=10 @ 52.9 | hsolve |  | 92.44 [90.77-92.60] | 93.05 | 4.49 | 24.46 (1) | 136 / 11.97 | - | 14.23 | 0.23 | 13.22 / 13.22 | yes (3/3) | complete; ref 3e-17; slepc 1e-14 |
| yee-pec-box-1e6 | 1027733 | GHEP | k=10 @ 52.9 | hsolve-auto |  | 65.49 [64.64-66.31] | 66.12 | 4.51 | 23.54 (1) | 258 / 30.24 | - | 14.22 | 0.22 | 14.00 / 14.00 | yes (3/3) | complete; ref 3e-17; slepc 1e-14 |
| yee-pec-box-1e6 | 1027733 | GHEP | k=10 @ 52.9 | hsolve-nocert |  | 43.48 [42.91-43.88] | 44.25 | 4.49 | 23.15 (1) | 136 / 11.97 | - | 14.23 | 0.22 | 12.97 / 12.97 | yes (3/3) | complete; ref 3e-17; slepc 1e-14 |
| yee-pec-box-1e6 | 1027733 | GHEP | k=10 @ 52.9 | slepc | `4x4/core eps_nev=k+2 st_mat_mumps_cntl_7=1e-10 st_mat_mumps_icntl_28=2 st_mat_mumps_icntl_29=2 st_mat_mumps_icntl_35=2` (case profile yee-pec-box-1e6) | 68.27 [59.80-68.31] | 71.90 | 6.72 | 24.40 (1) | 78 / 35.48 | 1.18 | - | - | 15.38 / 15.22 | yes (3/3) | complete; ref 2e-14 |

Speedup vs SLEPc (median solver time; wall in parentheses), given only when both solvers were valid in every repetition. 'vs class profile' / 'vs untuned default': SLEPc did not run a profile tuned for this case, so this is no comparison with the tuned baseline.

| case | hsolve | hsolve-auto | hsolve-nocert |
|---|---|---|---|
| syn-mem27-100 | 1.15x (1.18x) | 1.73x (1.76x) | 2.33x (2.37x) |
| yee-pec-box-1e6 | 0.74x (0.77x) | 1.04x (1.09x) | 1.57x (1.62x) |

Parts (one compare.py call each; all solvers of a case ran on the same node): cmp-p9-1e6-syn-mem27-100 on c005: syn-mem27-100; cmp-p9-1e6-yee-pec-box-1e6 on c005: yee-pec-box-1e6.
