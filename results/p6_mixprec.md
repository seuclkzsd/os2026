# P6 "mixprec": a single-precision factorization in the 16-core shift-invert solver

> Negative result kept on main for the record (2026-10-15). The code, benches and scripts this report names are on the unmerged branch `p6-mixprec-fix` (GitHub origin), not on main.


**Question.** Does a single-precision (smumps) factorization of the shift-invert matrix
K = A - sigma B make hsolve faster and leaner at 16 cores, at equal final accuracy and with the
certificate intact?

**Answer: no, not as a speed option. The gate fails and stage B was not started.** At most one of
the five head-to-head cases is projected at >= 1.10x: syn-mem27-46, at 1.11x in stage A and 1.09x
in a 5-round rerun (section 7.1), with a refinement that just meets the tolerance. Two variants
were projected from measured costs:

* **V1:** a single factor, refined to double accuracy in every operator application. This is the
  existing `factor_precision: single` path. It is 0.35-0.59x end to end (measured), and it fails
  outright on 3 of 10 cases.
* **V2:** a single factor without refinement during the Krylov phase, with a final refinement of
  the Ritz pairs. Its final refinement reaches the tolerance on only 3 of 8 cases. Guard vectors
  add a fourth (yee-bloch-1e4), where V2 is then projected at 0.84x.

Even if that refinement cost nothing, V2 would give 1.12-1.19x on the three 3-D head-to-head cases
and 0.90-0.99x on the two 2-D ones. **Recommendation: keep `factor_precision = double` as the only
default.**

What single precision does buy is memory: the factors take 0.51-0.67x of the double storage. On
the 1e6-unknown 3-D case this is 8.0 instead of 15.6 GB. A single factor would fit in-core in the
default 14 GB budget, where the double path goes out of core and takes 2.2x longer.

## Summary

* **Factorization:** smumps is 1.42-1.94x faster on the 3-D cases (2.03x at n = 1e6) and 1.07-1.17x on
  the 2-D and low-fill ones. The factor's memory is 0.51-0.53x on 3-D and 0.60-0.67x on low fill.
  * The inertia nu(sigma) of the single factor equals double on 8 of 10 cases.
  * At the two near-singular shifts it is off by one: the GHEP's single factor without detection
    is "valid" with 64992 negative pivots instead of 64993, and the complex one is invalid.
  * A single factor's count therefore must not be used for the certificate (as the driver already
    ensures).
* **Solves:** a single solve is only 1.14-1.35x faster on 3-D (0.99x for 1 RHS on the small
  yee-bloch-1e4), and 0.79-0.99x (slower) on 2-D and low fill. The
  BLAS hook (P5c, route `all`) speeds up only the d* BLAS of dmumps; smumps' s* calls go to plain
  OpenBLAS. With the hook off, a single solve is 1.69-1.79x faster than a double one.
* **Accuracy of a plain single solve:** relative residual ||r||/||Bx|| of 1.3e-7 to 6.7 at the driver's
  shifts, which are near eigenvalues by design, so eps_single * cond(K) ~ 1e-3..1.
  * Iterative refinement needs 1-5 extra solves per application to reach double level, and does not
    converge on syn-mem27-46 or syn-ghep-near-singular-real-1e5.
  * An accurate application therefore costs 2.7-16.6x a double one.
  * On syn-near-singular-complex-1e5 the single factor cannot be used at all. Its embedding splits
    an eigenvalue pair at the shift, so it is flagged singular even with detection off.
* **V1:** the shift-invert Lanczos with the refined single operator takes about the same number of
  applications but 3.9-7.4x the Krylov time.
  * End to end, the existing `--precision single` is 0.35-0.59x on the 7 cases where it is valid.
  * It is invalid on 3 cases:
    * syn-mem27-46: pairs with eta up to 2.8e-5 and a failed certificate in all 3 runs;
    * syn-ghep-near-singular-real-1e5: 1 pair;
    * syn-near-singular-complex-1e5: "exactly singular" error with no fallback.
* **V2:** the Krylov phase with the plain single operator converges in its own estimates.
  * At Krylov tolerance 1e-8 it takes 0.59-1.0x the applications and 0.44-1.15x the time of double.
  * At 1e-11 it takes 0.73-1.36x the applications.
  * Its Ritz pairs have backward errors of 1e-6 to 2e-3 (1e-12 on ss-tem152078).
  * The refinement (residual inverse iteration with the single factor plus Rayleigh-Ritz in
    double) reduces eta by only 0.45-0.83 per step after the first one.
  * That rate is the fixed-shift inverse-iteration rate of the contaminating far components, not a
    precision effect: refining with the double factor gives the same 0.69 on yee-pec-box-1e5, where
    0.69 = |89.53 - 52.85| / 52.85 against the curl-curl null space at 0.
  * Guard vectors (k + 4 or k + 8 refined pairs, section 7.1) lower the rate only where the
    contaminating eigenvalues are a few neighbours of the wanted ones.
    * yee-bloch-1e4: 0.62 to 0.36, then accepted in 11-12 steps, but projected at 0.84x.
    * ss-qa8fk-qa8fm: 0.66 to 0.51, still not accepted in 12 steps (18-20 steps with g = 8,
      projected at 0.69x).
    * Little or no change on yee-pec-box-1e5 (the curl-curl null space, far larger than any small
      g), syn-mem125-46 and cav2d-rect-pert-1e5.
    * The single-operator Krylov phase grows by a case-dependent 1.0x (syn-ghep) to 65x (cav2d).
  * A Davidson-type Rayleigh-Ritz over [X, S^-1 R] diverges, also with double corrections.
  * After 12 steps, 5 of 8 cases still miss the tolerance (eta 4e-10 to 3e-8).
* **Accuracy caveat:** on ss-tem152078 the single Ritz pairs pass eta <= 1e-10 without any refinement,
  but their eigenvalues differ from the double ones by up to 2.3e-8 relative to scale + |mu|.
  Backward error alone does not guarantee double-level eigenvalues there (ill-conditioned B). A
  single path would return less accurate results than double, which the task forbids.

## 1. Method

* **Where.** gpu02, measurement slot: cores 16-31, 16 threads, 32 GiB, `guard.py --exclusive`, one
  run at a time. Build and test lane: cores 8-15, used only for the convergence diagnostics of
  section 7, which quote no times (section 7.1 is measured in the slot). All runs were on 2026-10-13.
* **Base.** p5c-integration 1bd75f9: the BLAS hook linked with route `all`, `certificate = exact`,
  adaptive ordering. The bench is `bench_mixprec_phases`. The stage-A tables come from adf9cc9; the
  1e6 run comes from aa6edb7, which adds only the memory guard. Both are linked with the BLAS hook
  exactly like `hsolve` (CMake `hsolve_link_blas_hook`).
* **Baseline breakdown.** `tools/bench/compare.py` was run on the base binary, with 3 repetitions in
  one call. All 30 runs were valid by tools/eval at the case tolerance.
* **Stage A.** For each case the bench reads the report.json of baseline repetition 0: sigma_used,
  the resolved ordering, the factor cap and the block width p.
  1. It analyses K(sigma) with that ordering and factorizes once, which is the driver's numeric
     phase.
  2. It analyses a double object and a single object with that analysis' pivot order (PERM_IN), so
     both eliminate identically.
  3. It compares them over 5 interleaved repetitions with alternating order:
     * factorization;
     * 1- and p-RHS solves, 3 calls each per repetition;
     * accuracy of plain solves;
     * `ShiftInvertOperator` refinement, both to the double solve's ||r||/||Bx|| ("dlevel") and with
       the driver's V1 policy (rel_tol = tol * 0.1 * 0.1, at most 10 steps);
     * time per application.
  4. It runs the shift-invert Lanczos of the driver (nev = k, block p, Krylov tol = tol * 0.1) in
     3 rotated rounds:
     * double operator;
     * V1 (refined single operator);
     * V2, with the plain single operator at Krylov tolerances 1e-11 and 1e-8, followed by
       Rayleigh-Ritz in double and residual inverse iteration with the single factor until the k
       pairs nearest sigma have eta <= tol, at most 12 steps.
  5. The certificate is unchanged: double Sylvester counts. The single factor's inertia nu(sigma) is
     only recorded.
* **Projection.** It uses the 16-core baseline medians, and only the differences measured in the
  same bench process at 16 threads:
  * T_V = T_base - n_sigma * (numeric_double - numeric_single) + (iterate_V - iterate_double) + c.
  * n_sigma counts the factorizations at the working shift: 1, or 2 after a shift perturbation.
  * iterate = Lanczos plus Rayleigh-Ritz, plus V2's refinement.
  * c = the given-order analysis of an InertiaCounter, 0.02-0.34 s, for cases whose certificate
    counted on the main object, since a single main object cannot count exactly.
  * B's definiteness factorization and the certificate counts stay double and are unchanged.
  * A V2 that does not reach the tolerance within 12 steps is marked "fails": it would need a double
    refactorization and rerun, which is a loss.
  * V2 bound = T_base - n_sigma * (numeric_double - numeric_single) + (Krylov_V2 - Krylov_double) + c,
    with the faster of V2's two Krylov tolerances. It sets the Ritz-pair refinement and the extra
    Rayleigh-Ritz to zero and keeps c, like the two projected columns.
  * GHIEP (wg-square-1e5) has no Lanczos simulation. For it, V1 = base + solve time *
    (refined / double application cost), and the V2 bound = base with every solve at the plain
    single speed and free refinement. Its certificate is not performed (indefinite B), so c = 0.
* **Gate (from the task).** A variant must be projected >= 1.10x on at least three of the five
  head-to-head cases, with no loss above 3% on any.

## 2. Baseline (p6-base: compare.py, base 1bd75f9, 16 threads, medians of 3, all valid)

| case | total | setup | analysis | numeric (#) | solves (#/s) | iterate | verify | certificate | factor MB | peak GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| syn-mem27-46 | 2.53 | 0.04 | 0.62 | 0.48 (1) | 44 / 0.36 | 0.38 | 0.02 | 0.90 | 615 | 0.86 |
| yee-pec-box-1e5 | 2.95 | 0.02 | 0.57 | 0.51 (1) | 84 / 0.69 | 0.78 | 0.02 | 0.96 | 628 | 0.88 |
| wg-square-1e5 | 0.48 | 0.02 | 0.05 | 0.07 (1) | 62 / 0.20 | 0.30 | 0.04 | 0.00 | 90 | 0.21 |
| ss-qa8fk-qa8fm | 1.70 | 0.03 | 0.37 | 0.62 (3) | 100 / 0.42 | 0.52 | 0.01 | 0.40 | 297 | 0.60 |
| cav2d-rect-pert-1e5 | 0.31 | 0.02 | 0.03 | 0.10 (3) | 22 / 0.10 | 0.12 | 0.01 | 0.06 | 49 | 0.20 |
| syn-mem125-46 | 12.31 | 0.17 | 1.78 | 2.33 (1) | 144 / 3.26 | 3.37 | 0.03 | 4.49 | 2139 | 3.04 |
| ss-tem152078 | 5.46 | 0.20 | 1.35 | 2.39 (4) | 88 / 0.92 | 1.28 | 0.02 | 1.19 | 790 | 1.37 |
| yee-bloch-1e4 | 0.82 | 0.01 | 0.09 | 0.16 (1) | 68 / 0.17 | 0.19 | 0.00 | 0.33 | 183 | 0.30 |
| syn-near-singular-complex-1e5 | 2.47 | 0.07 | 0.70 | 0.59 (4) | 100 / 0.51 | 0.77 | 0.05 | 0.29 | 171 | 0.66 |
| syn-ghep-near-singular-real-1e5 | 1.29 | 0.05 | 0.49 | 0.26 (5) | 108 / 0.24 | 0.37 | 0.02 | 0.10 | 62 | 0.35 |

The numeric factorization at the working shift is 10-19% of the total on the head-to-head cases
(4-20% over all ten), and the solves are 14-42%. These are the only phases single precision
changes. The certificate counts stay double:
* InertiaCounter on syn-mem27-46, yee-pec-box-1e5, syn-mem125-46 and yee-bloch-1e4;
* main-object counts on the others.

## 3. Stage A: factorization, memory, solves (16 threads, medians of 5)

Same pivot order for both precisions. The table lists:
* MB used = INFOG(21), memory effectively used;
* RSS growth = resident-set increase at the first factorization;
* single detection = MUMPS null-pivot test at single_null_pivot_tol = 1e-4. Where it flagged the
  shift, the measurements ran without detection.

| case | n | type | ordering | p | numeric double / single s | factor MB used d / s | RSS growth GiB d / s | 1-RHS solve ms d / s | p-RHS solve ms d / s | single detection | nu(sigma) neg d / s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| syn-mem27-46 | 97336 | HEP | ND | 4 | 0.469 / 0.264 (**1.77x**) | 624 / 324 | 0.48 / 0.18 | 29.9 / 25.7 (1.16x) | 32.3 / 24.9 (1.30x) | yes | 1950 / 1950 |
| yee-pec-box-1e5 | 109744 | GHEP | SCOTCH | 4 | 0.500 / 0.279 (**1.79x**) | 646 / 333 | 0.44 / 0.17 | 31.3 / 26.5 (1.18x) | 34.2 / 26.3 (1.30x) | yes | 35478 / 35478 |
| wg-square-1e5 | 99225 | GHIEP | AMD | 2 | 0.055 / 0.047 (**1.16x**) | 90 / 54 | 0.05 / 0.02 | 6.3 / 6.6 (0.96x) | 5.8 / 5.9 (0.98x) | yes | 24659 / 24659 |
| ss-qa8fk-qa8fm | 66127 | GHEP | ND | 4 | 0.201 / 0.122 (**1.64x**) | 292 / 154 | 0.21 / 0.08 | 14.8 / 12.0 (1.24x) | 16.1 / 13.9 (1.16x) | yes | 79 / 79 |
| cav2d-rect-pert-1e5 | 99459 | GHEP | AMD | 1 | 0.032 / 0.027 (**1.17x**) | 49 / 32 | 0.02 / 0.01 | 4.8 / 6.0 (0.79x) | - | no (18 null) | 32900 / 32900 |
| syn-mem125-46 | 97336 | HEP | SCOTCH | 4 | 2.263 / 1.168 (**1.94x**) | 2083 / 1062 | 1.47 / 0.67 | 84.4 / 66.0 (1.28x) | 89.0 / 66.0 (1.35x) | yes | 1947 / 1947 |
| ss-tem152078 | 152078 | GHEP | ND | 4 | 0.587 / 0.414 (**1.42x**) | 784 / 413 | 0.46 / 0.17 | 38.2 / 31.7 (1.21x) | 41.7 / 31.1 (1.34x) | yes | 141072 / 141072 |
| yee-bloch-1e4 | 10080 | HEP (complex) | ND | 4 | 0.168 / 0.106 (**1.59x**) | 183 / 94 | 0.16 / 0.07 | 9.2 / 9.3 (0.99x) | 10.7 / 9.4 (1.14x) | yes | 3366 / 3366 |
| syn-near-singular-complex-1e5 | 100000 | HEP (complex) | ND | 4 | 0.144 / 0.135 (1.07x) | 171 / 109 | - | unusable | unusable | no (3 null); without detection the embedding splits a pair: singular | 65168 / 65167 (invalid) |
| syn-ghep-near-singular-real-1e5 | 100000 | GHEP | ND | 4 | 0.054 / 0.051 (1.07x) | 61 / 41 | 0.02 / 0.01 | 7.1 / 7.1 (0.99x) | 8.4 / 8.8 (0.95x) | no (1 null) | 64993 / 64992 |

ND = hsolve's nested dissection (ordering 10). The driver-path double factorization was one run
with the driver's own ordering, including the first call's allocations. It took 0.479, 0.518,
0.070, 0.215, 0.038, 2.287, 0.613, 0.164, 0.149 and 0.057 s. That is -2..+7% of the given-order
double medians, and +20-27% (at most 15 ms) on the two 2-D cases. The comparison therefore does
not flatter single.

**BLAS hook vs plain OpenBLAS** (`HS_BLAS_ROUTE=off`, same bench, medians of 5):

| case | numeric d / s | 1-RHS solve ms d / s | p-RHS solve ms d / s |
|---|---|---|---|
| syn-mem27-46 | 0.473 / 0.267 (1.77x) | 44.2 / 25.8 (**1.71x**) | 42.9 / 24.9 (**1.72x**) |
| yee-pec-box-1e5 | 0.492 / 0.267 (1.84x) | 43.3 / 25.6 (**1.69x**) | 41.7 / 24.4 (**1.71x**) |
| ss-qa8fk-qa8fm | 0.199 / 0.123 (1.62x) | 21.8 / 12.1 (**1.79x**) | 23.6 / 14.1 (**1.67x**) |

Without the hook, single halves the factor bytes the solve streams, and the solve is about 1.7x
faster. The hook speeds up the double solves by 1.3-1.5x, but smumps' s* calls never reach it, so
the hooked comparison shrinks to 1.16-1.35x. Float routes in the hook (s* narrow kernels) would be
needed to recover the bandwidth ratio. The factorization gain does not depend on the hook.

## 4. Accuracy of single solves and mixed-precision refinement (p columns)

| case | double solve rel / omega | single solve rel / omega | single forward error | IR steps to the double level | IR steps, V1 policy | application ms: double | single plain | single + IR (double level) | single + IR (V1) |
|---|---|---|---|---|---|---|---|---|---|
| syn-mem27-46 | 9.8e-09 / 8.5e-11 | **6.7** / 2.6e-02 | 0.26 | 20, NOT reached (2e-03) | 10, NOT reached (7e-03) | 32.9 | 25.5 | 546.5 (16.6x) | 300.6 (9.1x) |
| yee-pec-box-1e5 | 2.0e-12 / 6.1e-13 | 1.4e-03 / 5.0e-04 | 2.5e-04 | 4 | 4 | 34.8 | 26.5 | 142.2 (4.1x) | 141.7 (4.1x) |
| wg-square-1e5 | 3.4e-13 / 5.0e-13 | 2.8e-04 / 9.0e-04 | 8.2e-04 | 4 | 4 | 6.3 | 6.6 | 37.5 (6.0x) | 29.7 (4.7x) |
| ss-qa8fk-qa8fm | 5.4e-12 / 1.3e-11 | 9.0e-03 / 1.9e-02 | 7.2e-03 | 3 | 3 | 17.0 | 14.9 | 64.8 (3.8x) | 65.0 (3.8x) |
| cav2d-rect-pert-1e5 | 8.7e-13 / 1.5e-15 | 4.5e-04 / 5.5e-07 | 3.9e-04 | 3 | 3 | 4.9 | 6.2 | 25.5 (5.2x) | 25.7 (5.2x) |
| syn-mem125-46 | 7.6e-11 / 5.2e-11 | 1.8e-02 / 1.1e-02 | 9.1e-03 | 4 | 5 | 90.2 | 67.1 | 360.1 (4.0x) | 432.3 (4.8x) |
| ss-tem152078 | 3.4e-16 / 1.0e-06 | 1.3e-07 / 0.74 | 2.9e-03 | 3 | 1 | 44.6 | 34.2 | 121.8 (2.7x) | 79.0 (1.8x) |
| yee-bloch-1e4 | 4.0e-12 / 1.6e-12 | 1.7e-03 / 6.3e-04 | 6.9e-04 | 4 | 4 | 10.8 | 9.6 | 50.0 (4.6x) | 49.4 (4.6x) |
| syn-near-singular-complex-1e5 | - | single factor unusable | - | - | - | - | - | - | - |
| syn-ghep-near-singular-real-1e5 | 3.3e-09 / 3.6e-09 | 8.8e-03 / 2.4e-02 | **1.0** | 3, NOT reached (9e-03) | 3, NOT reached (9e-03) | 8.7 | 9.4 | 46.6 (5.3x) | 47.6 (5.4x) |

Each refinement step contracts the error by about eps_single * cond(K) * growth, and a
shift-invert shift puts that product at 1e-3 to above 1. At the two near-singular shifts the single
factor is useless as a solver. Its forward error is 0.26 and 1.0, and the refinement diverges or
stagnates. ss-tem152078's componentwise omega is distorted by cancellation (sigma = 1.7e14, K small
next to A and sigma B): its normwise residual is the meaningful one.

## 5. Shift-invert Lanczos with single operators (16 threads, 3 rounds, medians)

V2 is the plain single operator at Krylov tolerance 1e-8, then Rayleigh-Ritz and residual inverse
iteration X <- X - S^-1 (A X - B X Lambda) with the single factor S, then Rayleigh-Ritz again. The
"rate" is eta's geometric reduction per step from step 2 on.

| case | double: applications / Krylov s | V1: appl. / Krylov s / max eta | V2: appl. / Krylov s | V2 eta after RR | V2 refinement steps / s | V2 final eta | rate per step | V2 at Krylov tol 1e-11: appl. / s |
|---|---|---|---|---|---|---|---|---|
| syn-mem27-46 | 44 / 0.38 | 40 / 2.73 / 3e-04 (not accepted) | 28 / 0.19 | 1e-03 | 3 / 0.14 | 7e-11 | fast (~5e-3) | 32 / 0.22 |
| yee-pec-box-1e5 | 84 / 0.82 | 84 / 3.23 / 8e-13 | 80 / 0.61 | 1e-04 | 12 / 0.53 | 2e-08 (not accepted) | 0.69 | 100 / 0.78 |
| ss-qa8fk-qa8fm | 100 / 0.51 | 100 / 2.03 / 2e-13 | 80 / 0.37 | 4e-04 | 12 / 0.29 | 2e-08 (not accepted) | 0.61 | 100 / 0.46 |
| cav2d-rect-pert-1e5 | 22 / 0.13 | 22 / 0.60 / 7e-15 | 22 / 0.15 | 1e-06 | 12 / 0.16 | 1e-09 (not accepted) | 0.83 | 30 / 0.20 |
| syn-mem125-46 | 148 / 3.44 | 164 / 18.16 / 8e-14 | 88 / 1.53 | 8e-04 | 12 / 1.23 | 5e-10 (not accepted) | 0.45 | 108 / 1.88 |
| ss-tem152078 | 88 / 1.29 | 88 / 9.51 / 1e-16 | 72 / 0.87 | 1e-12 | 0 / 0 | 1e-12 (eigenvalues off by 2e-8) | - | 88 / 1.06 |
| yee-bloch-1e4 | 68 / 0.20 | 68 / 0.87 / 3e-12 | 60 / 0.16 | 1e-04 | 12 / 0.18 | 2e-08 (not accepted) | 0.64 | 76 / 0.21 |
| syn-ghep-near-singular-real-1e5 | 108 / 0.39 | 108 / 1.76 / 4e-04 (not accepted) | 88 / 0.33 | 2e-06 | 2 / 0.06 | 6e-12 | - | 108 / 0.40 |
| syn-mem27-100 (sec. 8) | (27.9 s iterate in hsolve, 148 appl.) | - | 90 / 11.50 | 2e-03 | 12 / 9.32 | 3e-08 (not accepted) | 0.55 | - |

The single operator's noise does not stop the Lanczos from converging in its own estimates. With a
looser Krylov tolerance it even uses fewer applications than double. But its Ritz pairs are those of
a perturbed operator, at eta = 1e-6..2e-3, and the cost moves into the refinement.

Eigenvalue agreement with the double arm, |lambda - mu| / (scale + |mu|):
* syn-mem27-46: <= 1e-18;
* syn-ghep-near-singular-real-1e5: 8e-16;
* the unconverged V2 runs: 1e-16..2e-12;
* ss-tem152078: 4e-9..2.3e-8 at eta 7e-13..1.6e-12, while the double arm reaches eta 6e-21.

## 6. Projection and gate

Projected 16-core totals in seconds; the speedup is base / projected. "V2 bound" assumes that the
Ritz-pair refinement and the extra Rayleigh-Ritz cost nothing. Like the projected columns, it
includes c: 0.04, 0.02, 0.14 and 0.10 s on ss-qa8fk-qa8fm, cav2d-rect-pert-1e5, ss-tem152078 and
syn-ghep-near-singular-real-1e5, whose certificates counted on the main object.

| case | base | numeric saved | V1 projected | V1 measured end to end (sec. 6.1) | V2 projected | V2 bound | V2 detail |
|---|---|---|---|---|---|---|---|
| **syn-mem27-46** | 2.53 | 0.20 | 4.67 (0.54x), not accepted | invalid (0/3) | **2.27 (1.11x)**; 2.31 (1.09x) in sec. 7.1 | 2.13 (1.19x) | 28 appl. 0.19 s + refine 0.14 s (3 steps) vs 44 / 0.40 s; final eta 5e-11..9e-11 (no margin); 3-4 steps in the 5 rounds of sec. 7.1 |
| **yee-pec-box-1e5** | 2.95 | 0.22 | 5.14 (0.57x) | 5.01 (0.59x) | fails | 2.52 (1.17x) | eta 1-2e-8 after 12 steps |
| **wg-square-1e5** (GHIEP) | 0.48 | 0.01 | 1.22 (0.39x) | 1.39 (0.35x) | <= 0.48 (0.99x) | 0.48 (0.99x) | single solves not faster (2-D) |
| **ss-qa8fk-qa8fm** | 1.70 | 0.08 | 3.17 (0.54x) | 2.89 (0.59x) | fails | 1.51 (1.12x) | eta 1-2e-8 after 12 steps |
| **cav2d-rect-pert-1e5** | 0.31 | 0.005 | 0.80 (0.38x) | 0.76 (0.40x) | fails | 0.34 (0.90x) | eta 1e-9 after 12 steps; single solve 0.79x |
| syn-mem125-46 | 12.31 | 1.09 | 25.93 (0.47x) | 25.27 (0.49x) | fails | 9.31 (1.32x) | eta 4-6e-10 after 12 steps (1.2 s) |
| ss-tem152078 | 5.46 | 0.17 | 13.65 (0.40x) | 15.24 (0.36x) | 5.01 (1.09x)* | 5.01 (1.09x) | *passes eta, but eigenvalues 2e-8 off double's |
| yee-bloch-1e4 | 0.82 | 0.06 | 1.42 (0.58x) | 1.42 (0.58x) | fails; 0.98 (0.84x) with g = 4 guards (sec. 7.1) | 0.72 (1.14x) | eta 2e-8 after 12 steps; with g = 4 accepted in 11-12 steps |
| syn-near-singular-complex-1e5 | 2.47 | - | - | error, no fallback (0/3) | fails | - | single factor unusable at sigma |
| syn-ghep-near-singular-real-1e5 | 1.29 | 0.01 | 2.76 (0.47x), not accepted | invalid (0/3) | 1.39 (0.93x) | 1.32 (0.97x) | low fill: nothing to gain |

**Gate: FAILED. Stage B was not started.**
* V1 loses on every case.
* V2 reaches >= 1.10x on at most 1 of 5 head-to-head cases: syn-mem27-46, with no accuracy margin,
  at 1.11x in stage A (3 refinement steps in each of 3 rounds) and 1.09x in the 5-round rerun of
  section 7.1 (4 steps in 3 of the 5 rounds). It fails on three, which would mean a double fallback
  and a loss, and is at <= 0.99x on wg-square-1e5. Guard vectors change none of this (section 7.1).
* An "auto" rule using single only on 3-D cases cannot rescue it: yee-pec-box-1e5 and
  ss-qa8fk-qa8fm fail the refinement.
* Even with a free refinement, only the three 3-D cases reach 1.12-1.19x. For 1.10x a refinement
  would have to cost at most 0.16 s on yee-pec-box-1e5 and 0.03 s on ss-qa8fk-qa8fm. At the
  measured 0.044 and 0.024 s per step, that is about 3 steps and 1 step. Measured with up to 30
  steps, ss-qa8fk-qa8fm needs 28-30 without guards and 18-20 with g = 8 (section 7.1).
  yee-pec-box-1e5, at 0.69 per step, would need about 26-29 (extrapolated from eta after 12 steps).

### 6.1 V1 end to end (the existing `--precision single`)

compare.py with a wrapper that appends `--precision single` to the base binary, 3 repetitions,
16 threads, all runs evaluated by tools/eval:

* **Valid on 7 of 10 cases**, at 0.35-0.59x of the base time, with peak RSS 0.76-1.06x of the base
  (smaller factors, plus the refinement workspace).
* **syn-mem27-46: 0 of 3 valid.** It needed 1063-1075 RHS instead of 44 and 14.6-15.1 s (medians
  1070 and 14.7 s). Over the 3 runs every returned pair has eta between 9e-9 and 2.8e-5 (tools/eval
  max_eta 2.82e-5), and the certificate fails: 6, 7 and 8 of 12 found. Every run reports
  success = false.
* **syn-ghep-near-singular-real-1e5: 0 of 3 valid.** One pair is returned, with rc = 1.
* **syn-near-singular-complex-1e5: rc = 2.** The error is "A - sigma B is exactly singular at and
  near sigma = 0.299994": the single factor is flagged singular at both shifts and the driver does
  not fall back to double.

## 7. Why the refinement is slow (diagnostics, build/test lane, convergence only)

These were run with the committed bench, on the baseline reports, `--reps 1 --no-v1 --v2-tols 1e-8`:

* **The double factor gives the same rate.** On yee-pec-box-1e5 with
  `--v2-refine-factor double --v2-mode rii`, eta goes 1e-4, 8e-6, 5e-6, ... 1e-7 after 12 steps:
  0.69 per step, identical to the single-factor refinement.
  * So the rate is that of fixed-shift inverse iteration on the contaminating components:
    |lambda_k - sigma| / |lambda_far - sigma|.
  * For the Yee box the far component is the curl-curl null space at lambda = 0:
    |89.53 - 52.85| / |0 - 52.85| = 0.694.
  * The Krylov phase sees the same ratio between the wanted theta = 1/36.7 and the null-space
    theta = -1/52.85, which is why double needs 84 applications there. A single solve injects
    null-space components at the 1e-4 level that only this ratio damps.
* **Guard vectors do not help on yee-pec-box-1e5.** `--v2-guards 0,4,8` refines k + g pairs: the
  rate stays 0.69-0.70, because the contamination is the curl-curl null space, far larger than any
  small g. In this single 8-thread run the single-operator Krylov phase needed 744 and 852
  applications for g = 4 and 8, against 80 for g = 0. Section 7.1 repeats the guard experiment on
  all eight V2 cases in the measurement slot: on other cases guards do lower the rate.
* **The Davidson-type extraction diverges.** `--v2-mode davidson` (Rayleigh-Ritz over
  [X, S^-1 R], keeping per old vector the Ritz vector of largest B-overlap) diverges on
  yee-pec-box-1e5 and ss-qa8fk-qa8fm: eta grows from 1e-4 to 1e-2 in 8 steps. It does so also with
  corrections from the double factor. On syn-mem27-46 it reaches only 5e-5. Standard Rayleigh-Ritz
  on the enlarged space admits spurious interior Ritz pairs.
* **Rates on the other cases** (stage A medians; the 5-round rerun of section 7.1 differs by up to
  0.06):
  * syn-mem125-46: 0.45, against the next eigenvalues beyond the 15 wanted;
  * ss-qa8fk-qa8fm: 0.61;
  * yee-bloch-1e4: 0.64;
  * cav2d-rect-pert-1e5: 0.83;
  * syn-mem27-100: 0.55.
  * Only syn-mem27-46, where the wanted 6-fold pairs are 1e3x nearer sigma than the rest, and
    syn-ghep-near-singular-real-1e5 converge in 2-4 steps.

Making V2 work would take a Krylov-accelerated refinement. One option is a few restarts of the
refined-operator Lanczos from the V2 basis, but each refined application costs 2.7-16.6x a double
one (section 4). The other is the existing Jacobi-Davidson with the single factor as
preconditioner, which P3 measured at 3-4x the double solves. Neither fits in the 0.03-0.16 s budget
above.

### 7.1 Guard vectors on every V2 case (measurement slot, 16 threads, 5 rounds)

`--v2-guards 0,4,8` carries g extra pairs through the Krylov phase (nev = k + g) and the
refinement. Acceptance stays on the k pairs nearest sigma. Each case ran in one bench process with
5 rotated rounds of the double arm and the three V2 arms. All V2 arms use Krylov tolerance 1e-8,
rii, and at most 12 steps; ss-qa8fk-qa8fm was run a second time with at most 30 steps.
* Applications are the range over the rounds; rate and times are medians.
* The rate is defined as in section 5.
* "Projected" uses the section 1 formula with this run's double arm and stage A's numeric saving
  and c, for arms accepted in all rounds, or in 4 of 5 rounds for the 30-step g = 0 arm.
* The g = 0 arms reproduce stage A's iterate times within 5%, except syn-mem27-46: its refinement
  took 4 steps instead of 3 in 3 of the 5 rounds (0.38 s instead of 0.34 s).

| case | Krylov applications: double / g = 0 / 4 / 8 | rate g = 0 / 4 / 8 | accepted rounds and final eta, g = 0 / 4 / 8 | iterate s: double / g = 0 / 4 / 8 | projected, g = 0 / 4 / 8 |
|---|---|---|---|---|---|
| syn-mem27-46 | 44 / 28 / 88-96 / 92-100 | 3-5 steps | 5/5 each, 2e-13..1e-10 | 0.40 / 0.38 / 0.85 / 0.99 | 2.31 (1.09x) / 2.77 (0.91x) / 2.92 (0.87x) |
| yee-pec-box-1e5 | 84 / 80-88 / 484-584 / 652-792 | 0.69 / 0.69 / 0.69 | 0/5 each, 3e-8..6e-8 | 0.82 / 1.18 / 5.12 / 6.40 | fails |
| ss-qa8fk-qa8fm | 100 / 80 / 104 / 104 | 0.66 / 0.57 / 0.51 | 0/5 each: 2e-8..3e-8 / 6e-9..1e-8 / 8e-10..5e-9 | 0.52 / 0.67 / 0.85 / 1.01 | fails |
| ss-qa8fk-qa8fm, at most 30 steps | 100 / 80 / 104 / 104 | 0.69 / 0.61 / 0.58 | 4/5 in 28-30 steps / 5/5 in 23-24 / 5/5 in 18-20 | 0.52 / 1.08 / 1.21 / 1.31 | 2.22 (0.76x) / 2.35 (0.72x) / 2.45 (0.69x) |
| cav2d-rect-pert-1e5 | 22 / 22 / 426-546 / 1094-1430 | 0.83 / 0.83 / 0.83 | 0/5 each, 9e-10..1e-9 | 0.14 / 0.31 / 3.95 / 10.95 | fails |
| syn-mem125-46 | 148 / 88-92 / 148-160 / 140-144 | 0.51 / 0.49 / 0.47 | 0/5 each: 5e-10 / 2e-10..5e-10 / 2e-10 | 3.34 / 2.66 / 3.92 / 3.82 | fails |
| ss-tem152078 | 88 / 72 / 84 / 92 | 0 steps | 5/5 each, 6e-13..1.3e-12 | 1.31 / 0.89 / 1.06 / 1.20 | 5.01 (1.09x)* / 5.18 (1.05x)* / 5.32 (1.03x)* |
| yee-bloch-1e4 | 68 / 60-64 / 76-80 / 128-132 | 0.62 / 0.36 / 0.35 | 0/5, 9e-9..2e-8 / 5/5 in 11-12 steps, 4e-11..1e-10 / same | 0.20 / 0.35 / 0.43 / 0.59 | fails / 0.98 (0.84x) / 1.14 (0.72x) |
| syn-ghep-near-singular-real-1e5 | 108 / 88 / 88 / 92 | 2 steps | 5/5 each, 3e-12..7e-12 | 0.39 / 0.39 / 0.43 / 0.51 | 1.39 (0.93x) / 1.43 (0.90x) / 1.51 (0.85x) |

\* eigenvalues off double's (section 5).

* **Guards help only where the contaminating components are a few eigenvalues next to the wanted
  ones**, which g extra pairs can carry along:
  * yee-bloch-1e4: g = 4 lowers the rate from 0.62 to 0.36, and V2 is accepted in 11-12 steps in
    every round. Its iterate is still 0.43 s against 0.20 s for double, i.e. 0.84x end to end.
  * ss-qa8fk-qa8fm: 0.66, 0.57 and 0.51 for g = 0, 4 and 8, not enough within 12 steps. With 30
    steps it is accepted after 28-30 (g = 0), 23-24 (g = 4) or 18-20 (g = 8) steps, at 0.69-0.76x.
* **Little or no change** on yee-pec-box-1e5 (0.69, the null space), cav2d-rect-pert-1e5 (0.83)
  and syn-mem125-46 (0.51, 0.49, 0.47).
* **The Krylov phase grows by a case-dependent factor**, from g = 0 to g = 4 or 8:
  * 1.0-1.3x on syn-ghep-near-singular-real-1e5, ss-tem152078, ss-qa8fk-qa8fm and yee-bloch-1e4
    (g = 4);
  * 1.5-2.2x on syn-mem125-46 and yee-bloch-1e4 (g = 8);
  * 3.1-3.6x on syn-mem27-46;
  * 5.5-10x on yee-pec-box-1e5;
  * 19-65x on cav2d-rect-pert-1e5.
  * With the noisy operator the count varies between rounds: 484-792 here on yee-pec-box-1e5,
    744 and 852 in the single 8-thread run above.
* **The gate is unchanged.** Where g = 0 already passes, no guard arm is faster than it. No guard
  arm passes on yee-pec-box-1e5 or cav2d-rect-pert-1e5. Where guards make V2 pass (yee-bloch-1e4,
  and ss-qa8fk-qa8fm with 30 steps), V2 is slower than double. A Krylov-accelerated refinement
  remains the only route.

## 8. The 1e6 case: memory (syn-mem27-100, n = 1e6, 27-point, sigma = 0, k = 12)

| run | numeric s | factor MB used (INFOG(21)) | MUMPS estimate MB | RSS growth GiB | solves | total s | peak GiB |
|---|---|---|---|---|---|---|---|
| hsolve base, budget 28 GB (in-core, SCOTCH) | 30.3 | 15948 | 18457 | 13.0 | 148 RHS / 26.4 s | 125.1 (valid) | 15.5 |
| hsolve base, default budget 14 GB (out-of-core) | 33.7 | 6198 (in core) | 18812 > budget | - | 152 RHS / 173.4 s | 274.6 (success) | 8.9 |
| single, same pivot order (bench) | **14.4** (2.03x) | **7976** (0.51x) | 9392 | 6.9 | 0.45 s / 1 RHS, 0.47 s / 4 RHS | - | - |

* **What single precision buys.** A single factor fits in-core in the default 14 GB budget, where
  the double path goes out of core and takes 2.2x longer (solves 6.6x slower).
* **Plain single solves here.** ||r||/||Bx|| = 0.075. Refinement needs 6 steps per application
  (V1 policy) or 8 (double level).
* **V2.** 90 applications, 11.5 s, against hsolve's in-core iterate of 27.9 s for 148. The
  refinement then stalls at eta 2-3e-8 after 12 steps (9.3 s, rate 0.55), so V2 does not solve
  this case either.
* **The simpler lever.** On a 32 GB machine, a larger memory budget (28 GB: in-core, 125 s) beats
  any single-precision path measured here.

## 9. Conclusion and recommendation

* **Keep `factor_precision = double` as the default and the only performance path.** The gate fails:
  * the factorization gain (1.4-1.9x on 3-D) is applied to 10-19% of the total;
  * the solve gain is 1.16-1.35x, or below 1 on 2-D, because the hook routes only d* BLAS;
  * the shift-invert shifts make K too ill-conditioned for a single factor to replace a double one
    without many extra solves.
* **The existing `factor_precision: single` (V1) is not a usable option today:**
  * 0.35-0.59x where it works;
  * wrong or no results on 3 of 10 cases;
  * on syn-near-singular-complex-1e5 it errors out instead of falling back.
  * The driver does not check `ShiftInvertOperator::last_relative_residual()` (DESIGN §3.4b).
  * If kept, it should be documented as a memory-only option, with that check and a double fallback.
* **Where mixed precision could still pay: memory-limited 3-D cases.** The factor is 0.51x, and the
  1e6 case at a 14 GB budget runs out of core in double. That needs a refinement that converges at
  more than inverse-iteration speed, which is a separate project.
* **Cheaper levers first:**
  * size the memory budget to the machine;
  * the certificate's single counts (certint `exact_single`, already in P5c).
* **Not done:** stage B (driver option, tests, end-to-end), because of the gate.

## Commands

Base binary: `build/release/hsolve` of 1bd75f9, copied to `$S/base/hsolve`
(`S=$HOME/p6_p6-mixprec` on gpu02). It runs from scratch repository copies: `tools/` copied,
`data` linked, `build/release/hsolve` linked to the binary, and for V1 a wrapper that appends
`--precision single`. `G = python3 tools/run/guard.py --exclusive --timeout 2700 --lock-wait 1200 --`.

```bash
# baseline (section 2) and V1 end to end (section 6.1)
$G python3 $R/tools/bench/compare.py --tag p6-base --cores 16-31 --solvers hsolve --reps 3 --mem-gb 28 \
   --timeout 600 --out-dir $S/cmp syn-mem27-46 yee-pec-box-1e5 wg-square-1e5 ss-qa8fk-qa8fm \
   cav2d-rect-pert-1e5 syn-mem125-46 ss-tem152078 yee-bloch-1e4 syn-near-singular-complex-1e5 \
   syn-ghep-near-singular-real-1e5                       # R = repo copy with the base binary
# same with --tag p6-v1 and R = repo copy whose build/release/hsolve appends --precision single
# stage A (sections 3-5), one slot call for all cases; bench built at adf9cc9
$G bash -c 'for c in CASES; do bench_mixprec_phases --case data/cases/$c \
   --report $R/results/raw/p6-base/$c/hsolve-r0/report.json --exp phases,v2 --reps 5 --inner 3 \
   --v2-mode rii --v2-tols 1e-11,1e-8 --v2-max-steps 12 > $S/stageA/$c.jsonl; done
 for c in syn-mem27-46 yee-pec-box-1e5 ss-qa8fk-qa8fm; do HS_BLAS_ROUTE=off bench_mixprec_phases \
   --case data/cases/$c --report ... --exp phases --reps 5 --inner 3 > $S/stageA/$c.routeoff.jsonl; done'
# 1e6 (section 8; data generated by results/gpu_g1.md's command into ~/os2026-gpu-data)
$G bash -c '$S/base/hsolve --A $C/A.npz --config $C/config.json --out $S/big/base28 -q --threads 16 \
   --memory-budget-gb 28; bench_mixprec_phases --case $C --report $S/big/base28/report.json --single-only \
   --exp phases,v2 --no-v1 --reps 3 --inner 2 --v2-mode rii --v2-tols 1e-8 --v2-max-steps 12 --max-factor-mb 26000'
python3 tools/run/guard.py --exclusive --timeout 1200 -- $S/base/hsolve --A $C/A.npz --config $C/config.json \
   --out $S/big/base14 -q --threads 16                  # default budget: out-of-core
# diagnostics (section 7), build/test lane, bench at 0550e3b
bench_mixprec_phases --case data/cases/yee-pec-box-1e5 --report ... --exp v2 --reps 1 --no-v1 --v2-tols 1e-8 \
   --v2-max-steps 12 --v2-refine-factor double --v2-mode rii
bench_mixprec_phases ... --v2-guards 0,4,8 --v2-max-steps 12 --v2-mode rii
bench_mixprec_phases --case data/cases/{syn-mem27-46,yee-pec-box-1e5,ss-qa8fk-qa8fm} ... --v2-max-steps 8 \
   --v2-mode davidson [--v2-refine-factor double]
# guard vectors on every V2 case (section 7.1): one slot call, bench built at 8aae708 (bench source as
# 0550e3b), CASES = the eight V2 cases of the section 7.1 table
python3 tools/run/guard.py --exclusive --timeout 2700 --lock-wait 1500 -- bash -c 'for c in CASES; do
   bench_mixprec_phases --case data/cases/$c --report $R/results/raw/p6-base/$c/hsolve-r0/report.json \
   --exp v2 --reps 9 --no-v1 --v2-tols 1e-8 --v2-guards 0,4,8 --v2-max-steps 12 --v2-mode rii \
   > $S/fix/guards/$c.jsonl; done                       # --reps 9 = 5 rotated rounds
 bench_mixprec_phases --case data/cases/ss-qa8fk-qa8fm ... (same) --v2-max-steps 30 \
   > $S/fix/guards/ss-qa8fk-qa8fm.steps30.jsonl'
```
