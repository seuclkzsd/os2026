# cmp4b

cores `28,29,40,41`, SLEPc ranks 1, wall/peak from guard.py

| case | n | type | k | hsolve wall/solve (s) | hsolve GiB | hsolve ok | slepc wall/solve (s) | slepc GiB | slepc ok | speedup vs slepc (wall / solve) | set check |
|---|---|---|---|---|---|---|---|---|---|---|---|
| wg-square-pert-1e4 | 9801 | GHIEP | 18 | 0.17 / 0.12 | 0.04 | yes | 2.61 / 0.11 | 0.14 | yes | 15.63x / 0.94x | hsolve:slepc 6e-13 |
| yee-pec-box-1e4 | 10064 | GHEP | 11 | 0.32 / 0.26 | 0.06 | yes | 2.68 / 0.39 | 0.17 | yes | 8.44x / 1.50x | hsolve:ref 1e-17; hsolve:slepc 2e-03; slepc:ref 2e-03 |
| ss-helm3d01 | 32226 | HEP | 10 | 1.01 / 0.94 | 0.14 | yes | 2.93 / 0.74 | 0.24 | yes | 2.91x / 0.79x | hsolve:slepc 8e-13 |
| ss-tem27623 | 27623 | GHEP | 10 | 1.01 / 0.90 | 0.14 | yes | 3.02 / 0.55 | 0.23 | yes | 2.98x / 0.61x | hsolve:slepc 4e-07 |
| cav2d-rect-pert-1e5 | 99459 | GHEP | 6 | 1.17 / 1.04 | 0.12 | yes | 3.28 / 0.84 | 0.25 | yes | 2.80x / 0.81x | hsolve:slepc 1e-13 |
| wg-square-1e5 | 99225 | GHIEP | 18 | 3.38 / 2.11 | 0.27 | yes | 4.53 / 1.83 | 0.33 | yes | 1.34x / 0.87x | hsolve:slepc 5e-13 |
| wg-halfslab-1e5 | 99225 | GHIEP | 28 | 2.84 / 2.73 | 0.35 | yes | 5.22 / 2.32 | 0.35 | NO | 1.84x / 0.85x | hsolve:slepc 1e-11 |
| syn-lap3d-aniso-1e5 | 97200 | HEP | 12 | 4.38 / 4.27 | 0.51 | yes | 5.42 / 3.19 | 0.54 | yes | 1.24x / 0.75x | hsolve:ref 5e-17; hsolve:slepc 3e-16; slepc:ref 3e-16 |
| syn-ghep-near-singular-real-1e5 | 100000 | GHEP | 10 | 2.21 / 1.79 | 0.20 | yes | 4.22 / 1.46 | 0.36 | yes | 1.91x / 0.81x | hsolve:ref 5e-16; hsolve:slepc 8e-16; slepc:ref 9e-16 |
| syn-zero-cluster-real-1e5 | 100000 | HEP | 21 | 4.23 / 3.97 | 0.34 | yes | 4.25 / 1.70 | 0.34 | yes | 1.00x / 0.43x | hsolve:ref 7e-17; hsolve:slepc 2e-15; slepc:ref 2e-15 |
| yee-pec-cube-32 | 92256 | HEP | 12 | 6.28 / 6.16 | 0.63 | yes | 6.22 / 3.89 | 0.72 | NO | 0.99x / 0.63x | hsolve:ref 9e-18; hsolve:slepc 2e-03; slepc:ref 2e-03 |
| yee-pec-box-1e5 | 109744 | GHEP | 10 | 6.33 / 6.18 | 0.78 | yes | 6.63 / 3.63 | 0.79 | NO | 1.05x / 0.59x | hsolve:ref 3e-17; hsolve:slepc 5e-04; slepc:ref 5e-04 |
| syn-mem27-46 | 97336 | HEP | 12 | 5.28 / 5.12 | 0.73 | yes | 6.04 / 3.65 | 0.84 | yes | 1.14x / 0.71x | hsolve:ref 2e-17; hsolve:slepc 4e-16; slepc:ref 4e-16 |
| ss-qa8fk-qa8fm | 66127 | GHEP | 10 | 3.73 / 3.53 | 0.46 | yes | 5.54 / 2.78 | 0.56 | yes | 1.49x / 0.79x | hsolve:slepc 1e-13 |
| ss-2cubes_sphere | 101492 | HEP | 10 | 6.45 / 6.25 | 0.67 | yes | 6.74 / 4.21 | 0.75 | NO | 1.04x / 0.67x | hsolve:slepc 3e-09 |
