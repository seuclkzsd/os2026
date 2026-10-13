# cmp-p8-best-interval

cores `16-31` (16), SLEPc profile `best`, 5 interleaved repetition(s): medians [min-max]. Solver time = hsolve time_total_s / SLEPc -log_view setup+solve+vectors (same scope: no process start-up, loading or output). Peak GiB: guard process tree / solver-reported. Valid = exit 0, success claimed and tools/eval pass at the case tol (completeness: complete = certified by a complete reference, consistent = agrees with an incomplete one); set errors |lambda-mu|/(|mu|+||A||/||B||) of the valid runs.

| case | n | type | target | solver | configuration | solver time (s) | wall (s) | analysis | numeric (#) | solves (#/s) | ortho | iterate | verify | peak GiB | valid | evaluation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yee-pec-box-1e5 | 109744 | GHEP | [40, 200] | hsolve |  | 3.84 [3.74-3.91] | 3.99 | 0.57 | 1.47 (3) | 144 / 1.20 | - | 1.51 | 0.13 | 1.21 / 1.21 | yes (5/5) | complete; ref 4e-17; slepc 2e-14 |
| yee-pec-box-1e5 | 109744 | GHEP | [40, 200] | hsolve-auto |  | 3.86 [3.76-3.89] | 4.05 | 0.56 | 1.50 (3) | 144 / 1.22 | - | 1.53 | 0.13 | 1.16 / 1.16 | yes (5/5) | complete; ref 4e-17; slepc 2e-14 |
| yee-pec-box-1e5 | 109744 | GHEP | [40, 200] | hsolve-nocert |  | 3.87 [3.75-3.91] | 4.02 | 0.57 | 1.49 (3) | 144 / 1.19 | - | 1.50 | 0.13 | 1.19 / 1.19 | yes (5/5) | complete; ref 4e-17; slepc 2e-14 |
| yee-pec-box-1e5 | 109744 | GHEP | [40, 200] | slepc | `4x4 eps_krylovschur_nev=60` (case profile yee-pec-box-1e5) | 10.88 [10.32-12.50] | 13.59 | 1.28 | 1.33 (2) | 194 / 7.11 | 0.94 | - | - | 1.94 / 1.87 | yes (5/5) | complete; ref 1e-13 |

Speedup vs SLEPc (median solver time; wall in parentheses), given only when both solvers were valid in every repetition. 'vs class profile' / 'vs untuned default': SLEPc did not run a profile tuned for this case, so this is no comparison with the tuned baseline.

| case | hsolve | hsolve-auto | hsolve-nocert |
|---|---|---|---|
| yee-pec-box-1e5 | 2.83x (3.40x) | 2.82x (3.36x) | 2.81x (3.38x) |
