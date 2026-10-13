# bench/

One executable per `bench/<name>.cpp` (target `bench_<name>`, built with the library when
`HSOLVE_BUILD_BENCH` is on), for reproducible performance and memory measurements that back numbers
quoted in docs and commit messages. Run them through `tools/run/guard.py` (e.g.
`python3 tools/run/guard.py --skip-busy-check --cores 26,27 --mem-gb 4 -- build/release/bench_<name> ...`);
they print machine-readable results (JSON lines) to stdout. The header comment of each source gives
the full usage and the exact command behind its quoted numbers; `results/p3_measurements.md` and
`results/p4_measurements.md` collect the P3 and P4 results. Heavier, case-based comparisons live in
`tools/bench/` (Python: SLEPc baseline `baseline_slepc.py`, tuner `tune_slepc.py`, `-log_view` parser
`slepc_log.py`, profiles `slepc_profile.py` / `slepc_profiles.json`, `compare.py`; see
`results/baseline_tuning.md`). `CASE` below is a directory under `$HS_DATA_ROOT/cases` (or
`./data/cases`). Since the end of P4 measurements run on the cluster nodes:
`tools/dev/cluster_run.sh gpu01 -- python3 tools/run/guard.py ... -- build/release/bench_<name> ...`
(core and memory defaults from `env/hosts/`; the core lists in the header comments are those of the
development server where the quoted numbers were taken).

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

P4 programs:

| program | measures | run |
|---|---|---|
| `bench_solvephase_profile` | time split of one shift-invert apply per p (B X, MUMPS solve, pack/unpack, refinement residual, apply with / without refinement), P1 vs P4 kernels in one process, backward errors, STREAM triad | `bench_solvephase_profile CASE_DIR [--ordering amd\|scotch\|metis\|auto] [--precision single] ...` |
| `bench_solvephase_mumps` | inside one MUMPS solve with the BLAS interposed: time above / inside the L0 layer (`--exp split`), ICNTL(27), ICNTL(21), solve threads, dgemm->dgemv for N = 1 (`gemvfwd`), parallel trsm, skinny GEMM | `bench_solvephase_mumps CASE_DIR --exp split\|icntl27\|icntl21\|partrsm\|gemvfwd\|skinny\|solvethreads [--p ...]` |
| `bench_solvephase_kernels` | spmm and `shifted_residual` per KernelPath (Reference / Portable / Avx512) on the case matrices, with the deviation from Reference | `bench_solvephase_kernels CASE_DIR [--p 1,2,4,8,16] [--reps 9]` |
| `bench_solvephase_lanczos` | shift-invert Lanczos with Reference vs Auto kernels at equal tolerance, per block size | `bench_solvephase_lanczos CASE_DIR [--p 1,8] [--reps 5]` |
| `bench_solvephase_fix_refine` | the three P4 refinement changes one at a time against a replica of the P1 operator (kernels, active-column solves, rounding-level stop) | `bench_solvephase_fix_refine CASE_DIR [--precision double\|single] [--p 1,8]` |
| `bench_solvephase_fix_lanczos` | the same Lanczos A/B with alternating arm order, kernel time apart from the solves, block size 1 as A/A control | `bench_solvephase_fix_lanczos CASE_DIR [--p 1,8] [--reps 9]` |
| `bench_analysis_orderings` | MUMPS analysis per ordering (ICNTL(7) or hsolve ND as PERM_IN): time, given-order split, estimates, factorization; `--exp separator` (multilevel trace), `--exp dissect` (work, critical path) | `bench_analysis_orderings --case DIR [--exp orderings\|separator\|dissect] --orderings 0,5,3,pnd:2 [--factorize 1]` |
| `bench_analysis_fix_pipeline` | what an ordering costs the driver: analysis, main factorization, refactorizations at the certificate ends or an InertiaCounter pair, 1- and p-RHS solves | `bench_analysis_fix_pipeline --case DIR --ordering 0\|3\|5\|10 [--counter 1]` |
| `bench_analysis_fix_imbalance` | MUMPS flops of hsolve's ND on grid Laplacians per separator imbalance (behind `kMaxImbalance`) | `bench_analysis_fix_imbalance [--grids ...] [--imbalance ...]` |
| `bench_drvint_profile` | a HEP/GHEP run outside the Krylov iteration: setup steps, AMD vs SCOTCH analyses with `model_analysis_s` (its fit data), certificate counts main object vs InertiaCounter (and `counter_pays`), per-RHS solve cost, output | `bench_drvint_profile --case DIR --exp setup,analysis,counter,solves,output` |
| `bench_drvint_fix_ordering` | AMD / SCOTCH / METIS analyses of K(sigma): time, fill and run-to-run stability, next to the driver's ordering decision | `bench_drvint_fix_ordering --case DIR [--orderings 0,3,5] [--factorize]` |
| `bench_ghiep_block` | block Krylov-Schur on GHIEP: `--mode ks` (apply cost per p, krylov_schur_general runs, accuracy), `--mode e2e` (hs::solve per `block_size`); synthetic pencils `syn-ghiep-<wg\|wg0\|cplx>-<n>` generated in process | `bench_ghiep_block [--mode ks\|e2e] [--p 1,2,4,8] CASE...` |
| `bench_ghiep_fix_e2e` | end-to-end GHIEP solves per block size with `--sigma` / `--k` overrides (the one-sided zero-cluster rows) | `bench_ghiep_fix_e2e [--p 1,2,4,8] [--sigma S] [--k K] CASE...` |
| `bench_ghiep_fix_ortho` | the block step's fast path (`block_cgs_cholqr2`) vs the two-pass `BlockOrthonormalizer` on the same inputs | `bench_ghiep_fix_ortho [--n 100000] [--k 24,48,72] [--p 2,4,8,16,64] [--input krylov\|random]` |
| `bench_certfast_measure` | exact certificate counts (driver path, InertiaCounter double / single pair) vs `exclusion_check` (delta x block size x pass radius), detection of removed copies, with hsolve's CLI solution as X | `bench_certfast_measure --case DIR --solution SOL [--reps 5] [--deltas 1e-6,1e-12] [--blocks 1,4]` |

Measurement scripts outside `bench/`: `python/bench/overhead.py` (Python API vs CLI wall time and
peak), `python/bench/load_memory.py` (`hsolve.load_matrix` vs `scipy.sparse.load_npz` memory),
`tools/eval/bench_norms.py` (the evaluator's norm-estimate settings).

P6 programs (mixed precision, `results/p6_mixprec.md`):

| program | measures | run |
|---|---|---|
| `bench_mixprec_phases` | double vs single (smumps) factorization of K = A - sigma B at the driver's shift, ordering and block width (from a baseline `report.json`): numeric time, factor memory, inertia, 1- and p-RHS solves, plain-single accuracy, mixed-precision refinement steps and time per accurate application (`--exp phases`); shift-invert Lanczos with the double, the refined single (V1) and the plain single operator plus Ritz-pair refinement (V2, `--exp v2`). Linked with the BLAS hook like `hsolve` | `bench_mixprec_phases --case DIR --report REPORT.json [--exp phases,v2] [--reps 5] [--v2-tols ...] [--v2-mode rii\|davidson] [--single-only]` |
