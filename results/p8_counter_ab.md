# P8: certificate counts on the InertiaCounter always vs the counter_pays rule

`inertia_counter = on` ("on") against the default rule ("auto": the counter where `counter_pays`, else
counts on the main factorization object) on 31 HEP/GHEP nearest cases, P8 binary (results/p8_certcost.md
§1), gpu01 measurement slot, 16 threads, ABBA order, 3 repetitions, medians of `time_total_s`; every
run valid (tools/eval, 186/186). "numeric #" counts the main object's factorizations (main-object
counts included). Section §3.4 of results/p8_certcost.md discusses it. Commands (from the work tree,
`S=$HOME/p8_p8-certcost`; the "on" binary is a wrapper that runs hsolve with the case config plus
`"inertia_counter": "on"`):

```bash
env/run.sh python3 tools/run/guard.py --exclusive --timeout 2700 -- python3 tools/bench/ab_cases.py run \
  --bin auto=$S/bin/hsolve_p8b --bin on=$S/bin/hsolve_p8b_cnton --reps 0,1,2 --run-dir $S/cnt-ab <31 cases>
env/run.sh python3 tools/run/guard.py --skip-busy-check --mem-gb 12 --timeout 3400 -- \
  python -m tools.eval.sweep $S/cnt-ab --out $S/cnt-ab/eval.md
env/run.sh python3 tools/bench/ab_cases.py table --base auto --new on --md out.md --json out.json $S/cnt-ab
```

| case | n | type | node | auto s [min-max] | on s [min-max] | on/auto | valid auto / on | numeric # auto / on | block p auto / on | certificate on |
|---|---|---|---|---|---|---|---|---|---|---|
| cav2d-rect-pert-1e5 | 99459 | GHEP | gpu01 | 0.317 [0.298-0.366] | 0.309 [0.308-0.312] | 0.973 | 3/3 / 3/3 | 3 / 1 | 2 / 2 | exact |
| cav2d-square-1e4 | 9976 | GHEP | gpu01 | 0.046 [0.046-0.052] | 0.053 [0.051-0.056] | 1.153 **slower** | 3/3 / 3/3 | 3 / 1 | 2 / 2 | exact |
| ss-2cubes_sphere | 101492 | HEP | gpu01 | 2.887 [2.874-2.916] | 2.879 [2.856-2.918] | 0.997 | 3/3 / 3/3 | 1 / 1 | 4 / 4 | exact |
| ss-helm3d01 | 32226 | HEP | gpu01 | 0.550 [0.549-0.568] | 0.565 [0.563-0.570] | 1.027 | 3/3 / 3/3 | 3 / 1 | 4 / 4 | exact |
| ss-qa8fk-qa8fm | 66127 | GHEP | gpu01 | 1.673 [1.671-1.685] | 1.665 [1.661-1.668] | 0.995 | 3/3 / 3/3 | 3 / 1 | 4 / 4 | exact |
| ss-tem152078 | 152078 | GHEP | gpu01 | 5.427 [5.412-5.433] | 5.404 [5.401-5.438] | 0.996 | 3/3 / 3/3 | 4 / 2 | 4 / 4 | exact |
| ss-tem27623 | 27623 | GHEP | gpu01 | 0.593 [0.588-0.637] | 0.600 [0.597-0.604] | 1.011 | 3/3 / 3/3 | 4 / 2 | 4 / 4 | exact |
| syn-ghep-multiplicity-real-1e4 | 10000 | GHEP | gpu01 | 0.138 [0.137-0.141] | 0.142 [0.141-0.158] | 1.026 | 3/3 / 3/3 | 4 / 2 | 4 / 4 | exact |
| syn-ghep-near-singular-real-1e5 | 100000 | GHEP | gpu01 | 1.290 [1.283-1.291] | 1.386 [1.385-1.394] | 1.074 **slower** | 3/3 / 3/3 | 5 / 3 | 4 / 4 | exact |
| syn-ghep-zero-cluster-complex-1e4 | 10000 | GHEP | gpu01 | 0.293 [0.290-0.347] | 0.314 [0.312-0.316] | 1.070 **slower** | 3/3 / 3/3 | 3 / 1 | 4 / 4 | exact |
| syn-lap2d-100 | 10000 | HEP | gpu01 | 0.062 [0.060-0.067] | 0.068 [0.065-0.070] | 1.090 **slower** | 3/3 / 3/3 | 3 / 1 | 4 / 4 | exact |
| syn-lap3d-22 | 10648 | HEP | gpu01 | 0.144 [0.141-0.146] | 0.154 [0.140-0.166] | 1.071 **slower** | 3/3 / 3/3 | 3 / 1 | 4 / 4 | exact |
| syn-lap3d-aniso-1e5 | 97200 | HEP | gpu01 | 2.074 [2.064-2.081] | 2.071 [2.070-2.072] | 0.998 | 3/3 / 3/3 | 1 / 1 | 4 / 4 | exact |
| syn-mem27-46 | 97336 | HEP | gpu01 | 2.369 [2.360-2.371] | 2.405 [2.379-2.426] | 1.015 | 3/3 / 3/3 | 1 / 1 | 8 / 8 | exact |
| syn-multiplicity-complex-1e4 | 10000 | HEP | gpu01 | 0.206 [0.203-0.208] | 0.231 [0.229-0.238] | 1.123 **slower** | 3/3 / 3/3 | 3 / 1 | 4 / 4 | exact |
| syn-multiplicity-real-1e4 | 10000 | HEP | gpu01 | 0.095 [0.093-0.105] | 0.102 [0.101-0.121] | 1.074 **slower** | 3/3 / 3/3 | 3 / 1 | 4 / 4 | exact |
| syn-near-singular-complex-1e4 | 10000 | HEP | gpu01 | 0.236 [0.235-0.238] | 0.257 [0.255-0.263] | 1.090 **slower** | 3/3 / 3/3 | 4 / 2 | 4 / 4 | exact |
| syn-near-singular-complex-1e5 | 100000 | HEP | gpu01 | 2.496 [2.460-2.503] | 2.826 [2.824-2.844] | 1.132 **slower** | 3/3 / 3/3 | 4 / 2 | 4 / 4 | exact |
| syn-near-singular-real-1e4 | 10000 | HEP | gpu01 | 0.109 [0.105-0.119] | 0.111 [0.110-0.114] | 1.021 | 3/3 / 3/3 | 4 / 2 | 4 / 4 | exact |
| syn-wide-range-complex-1e4 | 10000 | HEP | gpu01 | 0.197 [0.188-0.202] | 0.209 [0.207-0.212] | 1.058 **slower** | 3/3 / 3/3 | 3 / 1 | 4 / 4 | exact |
| syn-wide-range-real-1e4 | 10000 | HEP | gpu01 | 0.099 [0.091-0.099] | 0.099 [0.093-0.108] | 0.997 | 3/3 / 3/3 | 3 / 1 | 4 / 4 | exact |
| syn-zero-cluster-complex-1e4 | 10000 | HEP | gpu01 | 0.311 [0.310-0.319] | 0.322 [0.321-0.325] | 1.035 **slower** | 3/3 / 3/3 | 5 / 2 | 4 / 4 | exact |
| syn-zero-cluster-real-1e4 | 10000 | HEP | gpu01 | 0.130 [0.130-0.132] | 0.127 [0.127-0.144] | 0.979 | 3/3 / 3/3 | 5 / 2 | 4 / 4 | exact |
| syn-zero-cluster-real-1e5 | 100000 | HEP | gpu01 | 1.226 [1.220-1.227] | 1.296 [1.289-1.314] | 1.057 **slower** | 3/3 / 3/3 | 6 / 2 | 4 / 4 | exact |
| yee-bloch-1e4 | 10080 | HEP | gpu01 | 0.831 [0.822-0.858] | 0.835 [0.818-0.844] | 1.004 | 3/3 / 3/3 | 1 / 1 | 8 / 8 | exact |
| yee-pec-box-1e4 | 10064 | GHEP | gpu01 | 0.201 [0.198-0.203] | 0.208 [0.200-0.212] | 1.038 **slower** | 3/3 / 3/3 | 3 / 1 | 4 / 4 | exact |
| yee-pec-box-1e5 | 109744 | GHEP | gpu01 | 2.975 [2.953-3.036] | 2.937 [2.935-2.967] | 0.987 | 3/3 / 3/3 | 1 / 1 | 4 / 4 | exact |
| yee-pec-cube-16 | 10800 | HEP | gpu01 | 0.221 [0.207-0.233] | 0.215 [0.213-0.219] | 0.971 | 3/3 / 3/3 | 3 / 1 | 4 / 4 | exact |
| yee-pec-cube-32 | 92256 | HEP | gpu01 | 2.313 [2.250-2.350] | 2.332 [2.208-2.342] | 1.008 | 3/3 / 3/3 | 1 / 1 | 8 / 8 | exact |
| yee-photonic-rods-1e4 | 10125 | HEP | gpu01 | 0.910 [0.907-0.913] | 0.910 [0.906-0.911] | 1.000 | 3/3 / 3/3 | 1 / 1 | 8 / 8 | exact |
| yee-photonic-spheres-1e4 | 10125 | GHEP | gpu01 | 0.968 [0.956-0.969] | 0.987 [0.958-0.994] | 1.020 | 3/3 / 3/3 | 1 / 1 | 8 / 8 | exact |

Geometric mean of on/auto over the 31 cases valid in every run: **1.034**; faster by > 3%: 0; slower by > 3%: 13 (cav2d-square-1e4 1.153, syn-ghep-near-singular-real-1e5 1.074, syn-ghep-zero-cluster-complex-1e4 1.070, syn-lap2d-100 1.090, syn-lap3d-22 1.071, syn-multiplicity-complex-1e4 1.123, syn-multiplicity-real-1e4 1.074, syn-near-singular-complex-1e4 1.090, syn-near-singular-complex-1e5 1.132, syn-wide-range-complex-1e4 1.058, syn-zero-cluster-complex-1e4 1.035, syn-zero-cluster-real-1e5 1.057, yee-pec-box-1e4 1.038).
