# cmp-p9int-1e6

cores `16-31` (16), SLEPc profile `best`, 5 interleaved repetition(s): medians [min-max]. Solver time = hsolve time_total_s / SLEPc -log_view setup+solve+vectors (same scope: no process start-up, loading or output). Peak GiB: guard process tree / solver-reported. Valid = exit 0, success claimed and tools/eval pass at the case tol (completeness: complete = certified by a complete reference, consistent = agrees with an incomplete one); set errors |lambda-mu|/(|mu|+||A||/||B||) of the valid runs.

| case | n | type | target | solver | configuration | solver time (s) | wall (s) | analysis | numeric (#) | solves (#/s) | ortho | iterate | verify | peak GiB | valid | evaluation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| syn-mem27-100 | 1000000 | HEP | k=12 @ 0 | hsolve |  | 82.46 [82.37-83.80] | 83.21 | 3.04 | 21.55 (1) | 160 / 13.95 | - | 15.47 | 0.27 | 13.11 / 13.11 | yes (5/5) | complete; ref 1e-16; slepc 9e-16 |
| syn-mem27-100 | 1000000 | HEP | k=12 @ 0 | hsolve-auto |  | 57.65 [56.83-57.81] | 58.43 | 3.03 | 21.66 (1) | 260 / 27.74 | - | 16.14 | 0.26 | 13.86 / 13.86 | yes (5/5) | complete; ref 1e-16; slepc 9e-16 |
| syn-mem27-100 | 1000000 | HEP | k=12 @ 0 | hsolve-nocert |  | 42.37 [41.57-42.46] | 43.16 | 3.06 | 21.58 (1) | 168 / 14.59 | - | 16.20 | 0.26 | 12.93 / 12.93 | yes (5/5) | complete; ref 1e-16; slepc 9e-16 |
| syn-mem27-100 | 1000000 | HEP | k=12 @ 0 | slepc | `1x16 eps_ncv=3k` (case profile syn-mem27-100) | 116.00 [115.75-116.13] | 120.30 | 14.23 | 23.38 (1) | 105 / 76.02 | 2.00 | - | - | 12.70 / 12.51 | yes (5/5) | complete; ref 1e-15 |
| yee-pec-box-1e6 | 1027733 | GHEP | k=10 @ 52.9 | hsolve |  | 69.13 [68.91-69.26] | 69.81 | 2.72 | 17.83 (1) | 136 / 10.91 | - | 13.16 | 0.23 | 11.31 / 11.31 | yes (5/5) | complete; ref 3e-17; slepc 2e-14 |
| yee-pec-box-1e6 | 1027733 | GHEP | k=10 @ 52.9 | hsolve-auto |  | 55.04 [54.89-55.17] | 55.78 | 2.72 | 17.80 (1) | 258 / 27.32 | - | 13.19 | 0.22 | 12.36 / 12.36 | yes (5/5) | complete; ref 3e-17; slepc 2e-14 |
| yee-pec-box-1e6 | 1027733 | GHEP | k=10 @ 52.9 | hsolve-nocert |  | 34.97 [34.88-35.00] | 35.69 | 2.54 | 17.79 (1) | 136 / 10.85 | - | 13.12 | 0.22 | 11.28 / 11.28 | yes (5/5) | complete; ref 3e-17; slepc 2e-14 |
| yee-pec-box-1e6 | 1027733 | GHEP | k=10 @ 52.9 | slepc | `4x4/core eps_ncv=3k eps_nev=k+2 st_mat_mumps_cntl_7=1e-10 st_mat_mumps_icntl_28=2 st_mat_mumps_icntl_29=2 st_mat_mumps_icntl_35=2` (case profile yee-pec-box-1e6) | 59.30 [58.26-59.58] | 63.01 | 6.76 | 24.61 (1) | 59 / 26.84 | 0.74 | - | - | 15.95 / 15.79 | yes (5/5) | complete; ref 4e-14 |

Speedup vs SLEPc (median solver time; wall in parentheses), given only when both solvers were valid in every repetition. 'vs class profile' / 'vs untuned default': SLEPc did not run a profile tuned for this case, so this is no comparison with the tuned baseline.

| case | hsolve | hsolve-auto | hsolve-nocert |
|---|---|---|---|
| syn-mem27-100 | 1.41x (1.45x) | 2.01x (2.06x) | 2.74x (2.79x) |
| yee-pec-box-1e6 | 0.86x (0.90x) | 1.08x (1.13x) | 1.70x (1.77x) |

Parts (one compare.py call each; all solvers of a case ran on the same node): cmp-p9int-1e6-syn-mem27-100 on gpu01: syn-mem27-100; cmp-p9int-1e6-yee-pec-box-1e6 on gpu01: yee-pec-box-1e6.
