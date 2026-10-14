# cmp-p9int-best-interval

cores `16-31` (16), SLEPc profile `best`, 5 interleaved repetition(s): medians [min-max]. Solver time = hsolve time_total_s / SLEPc -log_view setup+solve+vectors (same scope: no process start-up, loading or output). Peak GiB: guard process tree / solver-reported. Valid = exit 0, success claimed and tools/eval pass at the case tol (completeness: complete = certified by a complete reference, consistent = agrees with an incomplete one); set errors |lambda-mu|/(|mu|+||A||/||B||) of the valid runs.

| case | n | type | target | solver | configuration | solver time (s) | wall (s) | analysis | numeric (#) | solves (#/s) | ortho | iterate | verify | peak GiB | valid | evaluation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yee-pec-box-1e5 | 109744 | GHEP | [40, 200] | hsolve |  | 3.30 [3.26-3.32] | 3.45 | 0.26 | 1.31 (3) | 144 / 1.14 | - | 1.45 | 0.13 | 1.31 / 1.31 | yes (5/5) | complete; ref 4e-17; slepc 2e-13 |
| yee-pec-box-1e5 | 109744 | GHEP | [40, 200] | hsolve-auto |  | 3.31 [3.28-3.34] | 3.48 | 0.26 | 1.31 (3) | 144 / 1.14 | - | 1.45 | 0.13 | 1.30 / 1.30 | yes (5/5) | complete; ref 4e-17; slepc 2e-13 |
| yee-pec-box-1e5 | 109744 | GHEP | [40, 200] | hsolve-nocert |  | 3.31 [3.28-3.32] | 3.53 | 0.26 | 1.31 (3) | 144 / 1.14 | - | 1.45 | 0.13 | 1.30 / 1.30 | yes (5/5) | complete; ref 3e-17; slepc 2e-13 |
| yee-pec-box-1e5 | 109744 | GHEP | [40, 200] | slepc | `4x4 eps_krylovschur_nev=60` (case profile yee-pec-box-1e5) | 10.36 [10.32-12.49] | 13.04 | 1.28 | 1.29 (2) | 194 / 6.69 | 0.92 | - | - | 1.86 / 1.79 | yes (5/5) | complete; ref 2e-13 |

Speedup vs SLEPc (median solver time; wall in parentheses), given only when both solvers were valid in every repetition. 'vs class profile' / 'vs untuned default': SLEPc did not run a profile tuned for this case, so this is no comparison with the tuned baseline.

| case | hsolve | hsolve-auto | hsolve-nocert |
|---|---|---|---|
| yee-pec-box-1e5 | 3.14x (3.78x) | 3.13x (3.75x) | 3.13x (3.70x) |

Parts (one compare.py call each; all solvers of a case ran on the same node): cmp-p9int-best-interval on gpu01: yee-pec-box-1e5.
