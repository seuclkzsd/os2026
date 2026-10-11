# bench/

One executable per `bench/<name>.cpp` (target `bench_<name>`, built with the library when
`HSOLVE_BUILD_BENCH` is on), for reproducible performance and memory measurements that back numbers
quoted in docs and commit messages. Run them through `tools/run/guard.py` (e.g.
`python3 tools/run/guard.py --skip-busy-check --cores 26,27 --mem-gb 4 -- build/release/bench_<name> ...`);
they print machine-readable results (JSON lines) to stdout. The header comment of each source gives
the full usage and the exact command behind its quoted numbers; `results/p3_measurements.md`
collects the P3 results. Heavier, case-based comparisons live in `tools/bench/` (Python: SLEPc
baseline, `compare.py`). `CASE` below is a directory under `$HS_DATA_ROOT/cases` (or `./data/cases`).

| program | measures | run |
|---|---|---|
| `bench_memio_load` | peak memory / result size of `hs::load_matrix` per input layout (npz csr/csc/coo, one triangle, Hermitian part, dtypes, MatrixMarket) on a generated 27-point matrix | `bench_memio_load [--grid 70] [--mtx-grid 45]` or `--files FILE...` |
| `bench_memio_fix_rows` | the same for inputs with 1-3 entries per row (diagonal / tridiagonal, n = 3e6) | `bench_memio_fix_rows [--n N] [--band B]...` |
| `bench_memio_pencil` | retained memory, construction and `assemble()` time of `ShiftedPencil`, MUMPS input size | `bench_memio_pencil --A A.npz [--B B.npz] [--reps R]` |
| `bench_memio_e2e` | memory profile of a whole CLI-equivalent run (VmRSS sampled every 2 ms, phases from the driver log); `--analyze-only` prints MUMPS estimates first | `bench_memio_e2e --A A.npz [--B B.npz] [--config config.json] [--analyze-only]` |
| `bench_ortho_kernels` | deterministic `dot`/`norm2` vs OpenMP reductions; `orthogonalize_cgs2` vs the legacy BLAS2 CGS2; block CGS2 vs column calls | `bench_ortho_kernels reduce <n>` \| `cgs <n> <k> [reps]` \| `block <n> <k> <p> [reps]` |
| `bench_ortho_fix_kernels` | follow-up kernel measurements, interleaved with rotated order: reductions, single-thread row-block kernels (array vs vector-extension form vs gemv), CGS at small n | `bench_ortho_fix_kernels reduce <n>` \| `kernels <n> <k>` \| `cgs <n> <k> [rounds]` |
| `bench_ortho_shift_invert` | time split (operator / orthogonalization / ...) of the driver's shift-invert Krylov path on a case, loss of orthogonality | `bench_ortho_shift_invert <case dir> [reps] [ncv] [nev]` |
| `bench_ortho_fix_steps` | the same at a fixed number of restarts (tol 1e-300), so that two library versions do identical work | `bench_ortho_fix_steps <case dir> <restarts> [reps] [nev] [ncv]` |
| `bench_certpar_inertia` | cost of the certificate's inertia counts: ICNTL(31), PERM_IN reuse, concurrency, single precision, block pair, `InertiaCounter` vs the driver path | `bench_certpar_inertia --case DIR --lower a --upper b --exp E[,E...]` |
| `bench_certpar_fix_probes` | MUMPS call lock exclusion (`--exp lock`, meaningful in the debug preset) and missed null pivots on exact eigenvalue shifts (`--exp nullpivot`) | `bench_certpar_fix_probes --exp lock` \| `--exp nullpivot [--nx 100 --ny 101]` |
| `bench_blockks_shiftinvert` | block Krylov-Schur (p = 1/2/4/8) in the real shift-invert path: per-block operator cost, iteration time, applications, accuracy | `bench_blockks_shiftinvert [--p 1,2,4,8] [--ncv 0,64] [--reps 5] CASE...` |
| `bench_blockks_fix_matched` | block vs single-vector Lanczos at matched accuracy (loosest block tol reaching p = 1's residual) | `bench_blockks_fix_matched [--reps 3] CASE...` |
| `bench_lobpcg_jd_memory` | memory-limited fallback study, one mode per process: shift-invert Lanczos (double / single + refinement), JD with a single / double factor, driver modes, LOBPCG with and without a preconditioner | `bench_lobpcg_jd_memory --case NAME --mode MODE [--sigma S]` |
| `bench_lobpcg_jd_fix_options` | Jacobi-Davidson extraction / start-iteration options with a fresh factor per run (distributions) | `bench_lobpcg_jd_fix_options --case NAME [--extraction ritz] [--repeat 5]` |
| `bench_lobpcg_jd_fix_synthetic` | Jacobi-Davidson robustness on small known spectra (target on an eigenvalue, multiplicity >= nev), with and without a preconditioner | `bench_lobpcg_jd_fix_synthetic [--start N] [--extraction both]` |
| `bench_slicing_sweep` | `hs::solve` on a case for several settings: interval slice sizes, smallest / largest (vs the pre-P3 path), nearest with zero-cluster handling | `bench_slicing_sweep --A A.npz [--B B.npz] interval LO HI SIZE...` \| `smallest K` \| `largest K` \| `nearest SIGMA K [ZFT]` |
| `bench_slicing_fix_probe` | regression probes of the slicing review (multiplets, clusters, null spaces, anchors, GHIEP zero cluster) on small known spectra | `bench_slicing_fix_probe PROBE\|all [SEED]` |
| `bench_pybind_convert` | cost of converting scipy/numpy buffers to `hs::CsrMatrix` (python/src/csr_convert.hpp) per layout and dtype, against an owning copy | `bench_pybind_convert [N=100] [repeats=5]` |

Measurement scripts outside `bench/`: `python/bench/overhead.py` (Python API vs CLI wall time and
peak), `python/bench/load_memory.py` (`hsolve.load_matrix` vs `scipy.sparse.load_npz` memory),
`tools/eval/bench_norms.py` (the evaluator's norm-estimate settings).
