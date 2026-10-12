# cmp-p16-best-interval

cores `16-31` (16), SLEPc profile `best`, 5 interleaved repetition(s): medians [min-max]. Solver time = hsolve time_total_s / SLEPc -log_view setup+solve+vectors (same scope: no process start-up, loading or output). Peak GiB: guard process tree / solver-reported. Valid = exit 0, success claimed and tools/eval pass at the case tol (completeness: complete = certified by a complete reference, consistent = agrees with an incomplete one); set errors |lambda-mu|/(|mu|+||A||/||B||) of the valid runs.

| case | n | type | target | solver | configuration | solver time (s) | wall (s) | analysis | numeric (#) | solves (#/s) | ortho | iterate | verify | peak GiB | valid | evaluation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yee-pec-box-1e5 | 109744 | GHEP | [40, 200] | hsolve |  | 4.42 [4.34-4.43] | 4.58 | 0.56 | 1.47 (3) | 144 / 1.57 | - | 2.08 | 0.13 | 1.18 / 1.18 | yes (5/5) | complete; ref 4e-17; slepc 2e-13 |
| yee-pec-box-1e5 | 109744 | GHEP | [40, 200] | hsolve-nocert |  | 4.34 [4.30-4.44] | 4.53 | 0.55 | 1.49 (3) | 144 / 1.57 | - | 2.07 | 0.13 | 1.19 / 1.19 | yes (5/5) | complete; ref 4e-17; slepc 2e-13 |
| yee-pec-box-1e5 | 109744 | GHEP | [40, 200] | slepc | `4x4 eps_krylovschur_nev=60` (case profile yee-pec-box-1e5) | 10.46 [10.38-12.83] | 13.20 | 1.28 | 1.34 (2) | 194 / 6.79 | 0.93 | - | - | 1.95 / 1.88 | yes (5/5) | complete; ref 2e-13 |

Speedup vs SLEPc (median solver time; wall in parentheses), given only when both solvers were valid in every repetition. 'vs class profile' / 'vs untuned default': SLEPc did not run a profile tuned for this case, so this is no comparison with the tuned baseline.

| case | hsolve | hsolve-nocert |
|---|---|---|
| yee-pec-box-1e5 | 2.37x (2.88x) | 2.41x (2.91x) |
