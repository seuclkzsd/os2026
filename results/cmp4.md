# cmp4

cores `28,29,40,41`, SLEPc ranks 1, wall/peak from guard.py

| case | n | type | k | hsolve wall/solve (s) | hsolve GiB | hsolve ok | hsolve-nocert wall/solve (s) | hsolve-nocert GiB | hsolve-nocert ok | slepc wall/solve (s) | slepc GiB | slepc ok | speedup vs slepc (wall / solve) | set check |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wg-square-pert-1e4 | 9801 | GHIEP | 18 | 0.32 / 0.26 | 0.05 | yes | 0.32 / 0.24 | 0.04 | yes | 2.37 / 0.12 | 0.14 | yes | 7.48x / 0.45x | hsolve:slepc 9e-13; hsolve-nocert:slepc 9e-13 |
| yee-pec-box-1e4 | 10064 | GHEP | 11 | 0.47 / 0.41 | 0.05 | yes | 0.37 / 0.33 | 0.05 | yes | 2.38 / 0.19 | 0.16 | yes | 5.09x / 0.46x | hsolve:ref 2e-17; hsolve:slepc 2e-03; hsolve-nocert:ref 2e-17; hsolve-nocert:slepc 2e-03; slepc:ref 2e-03 |
| ss-helm3d01 | 32226 | HEP | 10 | 1.57 / 1.47 | 0.11 | yes | 1.27 / 1.19 | 0.11 | yes | 2.88 / 0.73 | 0.24 | yes | 1.83x / 0.50x | hsolve:slepc 8e-13; hsolve-nocert:slepc 8e-13 |
| ss-tem27623 | 27623 | GHEP | 10 | 1.52 / 1.37 | 0.10 | yes | 1.81 / 1.14 | 0.10 | yes | 3.01 / 0.56 | 0.23 | yes | 1.98x / 0.41x | hsolve:slepc 4e-07; hsolve-nocert:slepc 4e-07 |
| cav2d-rect-pert-1e5 | 99459 | GHEP | 6 | 1.77 / 1.67 | 0.11 | yes | 1.61 / 1.49 | 0.11 | yes | 3.33 / 0.93 | 0.24 | yes | 1.88x / 0.56x | hsolve:slepc 1e-13; hsolve-nocert:slepc 1e-13 |
| wg-square-1e5 | 99225 | GHIEP | 18 | 3.37 / 3.25 | 0.29 | yes | 3.41 / 3.32 | 0.29 | yes | 4.48 / 1.81 | 0.33 | yes | 1.33x / 0.56x | hsolve:slepc 1e-01; hsolve-nocert:slepc 1e-01 |
| wg-halfslab-1e5 | 99225 | GHIEP | 28 | 5.65 / 5.55 | 0.40 | yes | 5.03 / 4.93 | 0.40 | yes | 5.18 / 2.33 | 0.36 | NO | 0.92x / 0.42x | hsolve:slepc 4e-11; hsolve-nocert:slepc 4e-11 |
| syn-lap3d-aniso-1e5 | 97200 | HEP | 12 | 6.03 / 5.94 | 0.43 | yes | 4.61 / 4.52 | 0.42 | yes | 5.49 / 3.26 | 0.54 | yes | 0.91x / 0.55x | hsolve:ref 5e-17; hsolve:slepc 4e-16; hsolve-nocert:ref 5e-17; hsolve-nocert:slepc 4e-16; slepc:ref 4e-16 |
| syn-ghep-near-singular-real-1e5 | 100000 | GHEP | 10 | 2.61 / 2.29 | 0.19 | yes | 2.44 / 2.10 | 0.18 | yes | 4.25 / 1.46 | 0.37 | yes | 1.63x / 0.64x | hsolve:ref 3e-16; hsolve:slepc 1e-15; hsolve-nocert:ref 3e-16; hsolve-nocert:slepc 9e-16; slepc:ref 9e-16 |
| syn-zero-cluster-real-1e5 | 100000 | HEP | 21 | 6.13 / 5.85 | 0.34 | yes | 2.62 / 2.33 | 0.22 | yes | 4.18 / 1.60 | 0.33 | yes | 0.68x / 0.27x | hsolve:ref 6e-17; hsolve:slepc 2e-15; hsolve-nocert:ref 1e-08; hsolve-nocert:slepc 1e-08; slepc:ref 2e-15 |
| yee-pec-cube-32 | 92256 | HEP | 12 | 8.59 / 8.47 | 0.55 | yes | 6.78 / 6.66 | 0.55 | yes | 6.03 / 3.77 | 0.72 | NO | 0.70x / 0.44x | hsolve:ref 6e-18; hsolve:slepc 2e-03; hsolve-nocert:ref 9e-18; hsolve-nocert:slepc 2e-03; slepc:ref 2e-03 |
| yee-pec-box-1e5 | 109744 | GHEP | 10 | 14.44 / 14.30 | 0.62 | yes | 12.35 / 12.20 | 0.62 | yes | 6.74 / 3.68 | 0.83 | NO | 0.47x / 0.26x | hsolve:ref 3e-17; hsolve:slepc 5e-04; hsolve-nocert:ref 3e-17; hsolve-nocert:slepc 5e-04; slepc:ref 5e-04 |
| syn-mem27-46 | 97336 | HEP | 12 | 7.79 / 7.64 | 0.64 | yes | 6.88 / 6.73 | 0.63 | yes | 6.02 / 3.61 | 0.84 | yes | 0.77x / 0.47x | hsolve:ref 2e-17; hsolve:slepc 4e-16; hsolve-nocert:ref 2e-17; hsolve-nocert:slepc 4e-16; slepc:ref 4e-16 |
| ss-qa8fk-qa8fm | 66127 | GHEP | 10 | 6.13 / 5.94 | 0.36 | yes | 5.18 / 5.00 | 0.35 | yes | 5.02 / 2.39 | 0.56 | yes | 0.82x / 0.40x | hsolve:slepc 9e-14; hsolve-nocert:slepc 9e-14 |
| ss-2cubes_sphere | 101492 | HEP | 10 | 8.49 / 8.28 | 0.55 | yes | 6.62 / 6.41 | 0.54 | yes | 5.59 / 2.91 | 0.74 | NO | 0.66x / 0.35x | hsolve:slepc 3e-09; hsolve-nocert:slepc 3e-09 |
