# cmp-p5c-best-interval

cores `16-31` (16), SLEPc profile `best`, 5 interleaved repetition(s): medians [min-max]. Solver time = hsolve time_total_s / SLEPc -log_view setup+solve+vectors (same scope: no process start-up, loading or output). Peak GiB: guard process tree / solver-reported. Valid = exit 0, success claimed and tools/eval pass at the case tol (completeness: complete = certified by a complete reference, consistent = agrees with an incomplete one); set errors |lambda-mu|/(|mu|+||A||/||B||) of the valid runs.

| case | n | type | target | solver | configuration | solver time (s) | wall (s) | analysis | numeric (#) | solves (#/s) | ortho | iterate | verify | peak GiB | valid | evaluation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yee-pec-box-1e5 | 109744 | GHEP | [40, 200] | hsolve |  | 3.81 [3.80-3.82] | 4.02 | 0.55 | 1.46 (3) | 144 / 1.20 | - | 1.50 | 0.13 | 1.19 / 1.18 | yes (5/5) | complete; ref 4e-17; slepc 3e-13 |
| yee-pec-box-1e5 | 109744 | GHEP | [40, 200] | hsolve-auto |  | 3.78 [3.71-3.81] | 3.98 | 0.56 | 1.45 (3) | 144 / 1.19 | - | 1.50 | 0.13 | 1.18 / 1.18 | yes (5/5) | complete; ref 4e-17; slepc 3e-13 |
| yee-pec-box-1e5 | 109744 | GHEP | [40, 200] | hsolve-nocert |  | 3.79 [3.73-3.82] | 3.98 | 0.55 | 1.46 (3) | 144 / 1.20 | - | 1.51 | 0.13 | 1.22 / 1.22 | yes (5/5) | complete; ref 4e-17; slepc 3e-13 |
| yee-pec-box-1e5 | 109744 | GHEP | [40, 200] | slepc | `4x4 eps_krylovschur_nev=60` (case profile yee-pec-box-1e5) | 10.30 [10.28-12.37] | 12.74 | 1.28 | 1.29 (2) | 194 / 6.64 | 0.91 | - | - | 1.80 / 1.73 | yes (5/5) | complete; ref 3e-13 |

Speedup vs SLEPc (median solver time; wall in parentheses), given only when both solvers were valid in every repetition. 'vs class profile' / 'vs untuned default': SLEPc did not run a profile tuned for this case, so this is no comparison with the tuned baseline.

| case | hsolve | hsolve-auto | hsolve-nocert |
|---|---|---|---|
| yee-pec-box-1e5 | 2.70x (3.17x) | 2.72x (3.20x) | 2.71x (3.20x) |

Parts (one compare.py call each; all solvers of a case ran on the same node): cmp-p5c-best-interval on c005: yee-pec-box-1e5.
