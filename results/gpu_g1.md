# G1: SPRAL SSIDS (CPU and one V100) vs hsolve's MUMPS backend

Question: is SPRAL SSIDS (multifrontal LDL^T with inertia, BSD-3, CUDA path) faster than hsolve's
MUMPS 5.8.2 backend for the shift-invert factorization, solves and inertia counts, at equal CPU
resources (16 cores), with its GPU path on one V100 - and is SSIDS CPU-only faster anywhere?
Measured on gpu02, 2026-10-13, branch `exp-gpu-ssids`; a measurement prototype, nothing integrated.

**Answer.**

- Inertia: identical to MUMPS in every factorization (408 factorizations in 204 measured processes
  over 9 cases, plus the 22 tuning processes; no zero pivot), and equal to the exact count where the
  case knows it (syn-mem27-46: 1950, syn-mem125-46: 1947, syn-mem27-100: 20003).
- Factorization with the GPU: with SSIDS' defaults the V100 gains at most 1.37x (syn-mem27-100)
  and loses on 6 of 9 cases. Tuned (`gpu_perf_coeff` 4, `OMP_PROC_BIND=FALSE`, see "Tuning") it
  is **>= 2x only on the two largest 3-D problems**: syn-mem125-46 2.3x (0.99 vs 2.30 s) and the
  1e6-unknown 27-point problem syn-mem27-100 2.9x (10.0 vs 28.9 s, but 15.7 of the card's 16 GB);
  on the 1e5 3-D cases of the head-to-head 0.92-1.85x (yee-pec-box-1e5 1.34x only with MUMPS'
  ordering), ss-tem152078 1.10x.
- Solves are SSIDS' weak point: 8-column block solves take 1.6-2.9x MUMPS' time on the CPU and
  1.8-2.8x with the tuned GPU (the GPU solve loops over the right-hand sides one by one), parity
  only on the two largest 3-D cases (1.10x / 0.96x); one-column solves are faster only with the
  tuned GPU on the 3-D cases (1.03-2.0x). SSIDS' METIS analysis takes 1.5-16x the driver's
  adaptive MUMPS analysis; a CUDA context costs 0.13-0.15 s.
- Projected hsolve 16-core totals (factorizations, solves, analysis and certificate counts scaled by
  the measured ratios): **slower than hsolve with every SSIDS variant on all five reference cases**,
  0.27x-0.94x (best: tuned GPU with MUMPS' ordering on syn-mem27-46, 2.93 vs 2.76 s).
- SSIDS CPU-only: faster factorizations only where it gets a better ordering than the driver's AMD
  (2-D cases, 1.26-1.54x; MUMPS with METIS gains the same), 0.55-1.05x on the 3-D cases, slower
  solves everywhere (0.34-0.61x for 8 columns), up to 1.9x MUMPS' host memory (28 vs 15 GiB on
  syn-mem27-100). No CPU-only win: projected 0.30-0.80x.
- **G2 (SSIDS GPU as a runtime-selected backend with MUMPS fallback) is not worth it** by the
  agreed criterion (>= 2x factorization on the 3-D cases): it holds only above ~5e11 flops per
  factorization, where the 16 GB card is already nearly full at n = 1e6, and the slower solves and
  analysis cancel the gain end to end at the competition's 1e5 scale.

## G0: building SPRAL with the CUDA path (gpu02, offline)

`tools/dev/build_spral.sh` (run in the build/test lane) builds SPRAL **2025.09.18** (the latest
release; source tarball sha256 `1358168ac95297049e4fc810e54a16e0c765796cfbaa156e09979e2620b7dae7`)
with meson 1.11.1 (`meson.pyz`, sha256 `05f25d74eab0d1a9b26f1d2fe482677b9c2b9543f974016c8102a5f96b2b73f5`)
into `$HOME/opt/spral-2025.09.18-cuda12.2`: CUDA 12.2 for sm_70, the conda env's gcc/g++/gfortran
13.4, OpenBLAS 0.3.34, METIS 5.1 (32-bit), no hwloc, `-O3 -march=native`. Status: **GPU and CPU
paths built and working** after these fixes (each one is in the script and its commit message):

| problem | cause | fix |
|---|---|---|
| meson did not find nvcc | meson has no environment variable for the CUDA compiler | `$CUDA_HOME/bin` on PATH |
| nvcc could not compile any `.cu` with the conda g++ 13 | nvcc 12.2's front end cannot parse gcc 13's `<type_traits>` (`__is_convertible`, `__remove_cv`); `-allow-unsupported-compiler` does not help | host compiler g++ 9.2 (`$HOME/local/usr/gcc_9.2.0`, the only g++ 7..12 on the node) via `NVCC_PREPEND_FLAGS=-ccbin=...` |
| conda LDFLAGS broke nvcc | meson passes LDFLAGS (`-Wl,...`) to nvcc | link flags per language (`-D<lang>_link_args`) |
| linking `libspral.so` through nvcc failed (`libmvec_nonshared.a` not found) | meson links a library with `.cu` objects via nvcc -> g++ 9.2, which reads the conda sysroot's (glibc 2.28) libm linker script | one-line patch: `link_language : 'cpp'` (conda g++ links, + `-lgfortran`); SPRAL has no relocatable device code |
| `libspral.so` needed `libmvec.so.1` (absent on CentOS 7, glibc 2.17) | gfortran vectorizes `exp`/`log` loops (scaling, matching) into glibc vector-math calls declared by the 2.28 sysroot | Fortran with `-fno-tree-loop-vectorize`; the script now fails if the library needs glibc > 2.17 |
| executables could not resolve libspral's dependencies at link time | the conda ld (`--sysroot`) ignores the library's RPATH | `-Wl,-rpath-link` |
| **GPU solves corrupted the heap** (`free(): invalid next size` in `gpu_subtree solve_fwd`), **GPU factorization failed with CUDA error 700** on yee-pec-cube-16 | bugs of the 2025.09.18 release: the GPU solves pass the empty section `x(r*ldx+1:r*ldx:n)`; a missing `__syncthreads` in `reorder.cu` (delayed pivots); a missing `__threadfence` in `assemble.cu` | upstream commit 836dc3e "Fix SSIDS on GPU" (2026-07-17, not in any release) applied as `tools/dev/patches/spral-836dc3e-fix-ssids-gpu.patch` |

`tools/dev/build_spral.sh --smoke 7` runs SPRAL's SSIDS C example without a GPU and on GPU 7
(correct solution and pivot order in both). The 2025.09.18 release *without* the upstream patch must
not be used on a GPU.

## Method

`bench/gpu_ssids.cpp` (target `bench_gpu_ssids`, CMake option `HSOLVE_WITH_SPRAL`, preset `spral`)
assembles K = A - sigma B at the case's target sigma exactly as the driver's MUMPS input
(`ShiftedPencil::assemble(1, -sigma)` on the upper pattern; the same arrays are the lower-triangle
CSC SSIDS takes), then times, per backend and in its own process: the analysis, F factorizations at
sigma (the first cold, the others warm), the inertia of each, 10 one-column and 5 eight-column solves
(medians), and checks the normwise backward error ||Kx-b||_inf / (||K||_inf ||x||_inf + ||b||_inf).
Variants (backend:ordering):

- `mumps:driver` - hsolve's backend as the driver configures it: `MumpsFactorization` with
  `factor_options_from` (16 threads, L0-OpenMP, ICNTL(8)=7 scaling, null-pivot detection), the
  driver's adaptive ordering (AMD probe + cost model, SCOTCH when nested dissection pays; the probe
  is part of the analysis time). This is the reference of every speedup.
- `mumps:metis` - the same with METIS (ICNTL(7)=5), for the equal-ordering comparison.
- `ssids-cpu:metis` / `ssids-gpu:metis` - SSIDS with its default METIS ordering, without / with GPU 7.
- `ssids-cpu:mumps` / `ssids-gpu:mumps` - SSIDS given `mumps:driver`'s pivot order (SYM_PERM) as user
  order: the same elimination order as the reference (that MUMPS analysis is not in SSIDS'
  `analyse` column but is added in the projection).

SSIDS options: threshold u = 0.01 (MUMPS' default CNTL(1) for symmetric matrices is also 0.01),
scaling 4 (Ruiz equilibration, the counterpart of MUMPS' ICNTL(8)=7), block a-posteriori pivoting
(default), `action` = true, nemin 32, min_gpu_work 5e9 flops, gpu_perf_coeff 1 (defaults), one
explicit NUMA region of 16 threads (+ CUDA device 0 for the GPU backend), `OMP_CANCELLATION=TRUE
OMP_PROC_BIND=TRUE` as SSIDS requires; MUMPS runs in hsolve's environment. CUDA: `cuda_init_s` =
`cuInit` + the device's primary context (which SSIDS' runtime then uses), timed apart but part of
an end-to-end run; GPU memory = device-wide NVML "used" sampled every 5 ms (GPU 7 had no other
process: checked before every GPU process). Host memory = VmHWM of the backend phase (reset after
loading, includes the resident matrices).

Every timed process ran in gpu02's measurement slot (`guard.py --exclusive`: cores 16-31 = NUMA
node 1, the socket of GPU 7, 32 GiB), driven by `tools/bench/gpu_g1.sh` (all variants of a case in
one slot call, variant order rotated per repetition, repetitions interleaved over the cases);
tables by `tools/bench/gpu_g1_report.py`.

## Tuning the GPU path (before the main runs)

One repetition in the slot, 3 factorizations each (warm = median of the last two, s); data in
`results/gpu_g1_tuning.jsonl`. MUMPS (`mumps:driver`): 0.50 / 0.48 s.

| SSIDS configuration | syn-mem27-46 | yee-pec-box-1e5 | note |
|---|---|---|---|
| CPU only (`ssids-cpu:metis`) | 0.49 | 0.90 | same with OMP_PROC_BIND=FALSE (0.49 / 0.90) |
| GPU, defaults | 0.51 | 0.70 | |
| GPU mode, no subtree on the GPU (min_gpu_work 1e18) | 0.98 | 1.24 | CPU work at half speed in GPU mode |
| GPU, OMP_PLACES = the 16 cores listed twice | 0.42 | 0.57 | but the 8-RHS solve 8x slower |
| GPU, gpu_perf_coeff 2 / 4 / 8 (bound) | - / 0.33 / 0.55 | - / 0.69 / 0.70 | |
| GPU, gpu_perf_coeff 2 / 4, OMP_PROC_BIND=FALSE | 0.38 / **0.26** | 0.54 / **0.53** | chosen: "ssids-gpu tuned" |
| GPU, gpu_perf_coeff 4, min_gpu_work 1e9 | 0.32 | 0.71 | same as 5e9 |

Why: in GPU mode SSIDS starts an outer OpenMP team of two threads (CPU region + GPU driver) with
`proc_bind(spread)`, which splits the 16 places into two halves; the 16-thread CPU team then runs in
its half, two threads per core, while the GPU driver's half idles (the "no subtree on the GPU" row:
the same CPU work as the CPU-only run takes twice as long). OMP_PROC_BIND=FALSE (SSIDS then only
warns, flag +50) lets the OS spread the 17 threads; gpu_perf_coeff 4 (the V100 counted 4x a 16-core
region) moves more leaf subtrees to the GPU. The CPU-only path is insensitive to the binding. The
main runs report SSIDS GPU both with its defaults and tuned (`--gpu-perf-coeff 4`,
`OMP_PROC_BIND=FALSE`).

## Environment and commands

gpu02: 2x Xeon Silver 4314 (CentOS 7.9), timed runs on cores 16-31 (NUMA node 1) with <= 32 GiB;
GPU 7 = Tesla V100-PCIE-16GB (16384 MiB) on the same socket, driver 535.54.03, CUDA 12.2. Conda env
`os2026`: gcc/gfortran 13.4, MUMPS 5.8.2 (sequential, OpenMP), OpenBLAS 0.3.34 (OpenMP), METIS 5.1,
SCOTCH 7; the OpenMP runtime is LLVM's libomp (the env's libgomp.so.1 points to it). hsolve: branch
`exp-gpu-ssids` (main 2938cb4 + this experiment), preset `spral` (Release, -march=native). SPRAL
2025.09.18 + upstream 836dc3e as in G0. Every command below ran through `tools/dev/cluster_run.sh
gpu02 --` (each timed process one `guard.py --exclusive` run in the slot):

    tools/dev/build_spral.sh                       (inside guard.py in the build lane)
    tools/dev/build.sh spral
    tools/bench/gpu_g1.sh --reps 3 --out results/gpu_g1.jsonl syn-mem27-46 yee-pec-box-1e5 \
        wg-square-1e5 ss-qa8fk-qa8fm cav2d-rect-pert-1e5 syn-mem125-46 ss-tem152078 wg-square-1e6
    tools/bench/gpu_g1.sh --reps 3 --out results/gpu_g1.jsonl --variants ssids-gpu:metis,ssids-gpu:mumps \
        --bench-args "--gpu-perf-coeff 4" --ssids-env "OMP_PROC_BIND=FALSE" (same cases)
    tools/bench/gpu_g1.sh --reps 3 --out results/gpu_g1.jsonl --variants mumps:driver,ssids-cpu:metis,ssids-gpu:metis \
        --bench-args "--memory-budget-gb 30 --solves 5 --block-solves 3" ~/os2026-gpu-data/cases/syn-mem27-100
        (and the tuned ssids-gpu:metis with the same arguments)
    python3 tools/bench/gpu_g1_report.py results/gpu_g1.jsonl

(the default and tuned invocations were run with `--reps 1`, `2`, `3` alternately; the script skips
what the file already holds, so the repetitions of both interleave.) syn-mem27-100 (27-point stencil
on 100^3 points, n = 1e6, 2% negative, sigma = 0) was generated with `python -m tools.gen.synthetic
syn-mem27-100 --root ~/os2026-gpu-data`, outside the shared data/ so that no other run picks it up;
the analysis estimates were checked in the lane first (MUMPS: 1.4e9 factor entries, 19.4 GB
in-core; SSIDS: 1.2e9 entries). The driver's default 14 GB budget would cap MUMPS (ICNTL(23)) below
that, hence `--memory-budget-gb 30`; only four variants ran on it.

## Results (medians of 3 interleaved repetitions)

Raw data: `results/gpu_g1.jsonl` (204 processes), tuning `results/gpu_g1_tuning.jsonl`; the
complete per-variant table is the appendix. "tuned" = SSIDS GPU with `gpu_perf_coeff` 4 and
`OMP_PROC_BIND=FALSE`. Flops per factorization (MUMPS estimate / SSIDS count): syn-mem27-46
7.4e10 / 6.5e10, yee-pec-box-1e5 6.9e10 / 7.5e10, wg-square-1e5 1.2e9 / 7.7e8, ss-qa8fk-qa8fm
2.2e10 / 1.9e10, cav2d-rect-pert-1e5 1.7e8 / 2.8e8, syn-mem125-46 4.8e11 / 4.7e11, ss-tem152078
7.1e10 / 7.0e10, wg-square-1e6 5.5e10 (AMD) / 2.0e10 (METIS), syn-mem27-100 9.3e12 / 7.5e12.

**Factorization, warm (s); speedup vs `mumps:driver` in parentheses**

| case | n | driver ordering | mumps:driver | mumps:metis | ssids-cpu:metis | ssids-cpu:mumps | ssids-gpu:metis | ssids-gpu:mumps | tuned gpu:metis | tuned gpu:mumps |
|---|---|---|---|---|---|---|---|---|---|---|
| syn-mem27-46 | 97336 | SCOTCH | 0.494 | 0.467 (1.06) | 0.499 (0.99) | 0.494 (1.00) | 0.492 (1.00) | 0.510 (0.97) | 0.267 (**1.85**) | 0.285 (1.74) |
| yee-pec-box-1e5 | 109744 | SCOTCH | 0.489 | 0.523 (0.93) | 0.895 (0.55) | 0.790 (0.62) | 0.684 (0.71) | 0.620 (0.79) | 0.532 (0.92) | 0.365 (**1.34**) |
| wg-square-1e5 | 99225 | AMD | 0.059 | 0.048 (1.23) | 0.038 (**1.54**) | 0.058 (1.03) | 0.081 (0.73) | 0.132 (0.45) | 0.041 (1.45) | 0.059 (1.01) |
| ss-qa8fk-qa8fm | 66127 | SCOTCH | 0.221 | 0.209 (1.05) | 0.211 (1.05) | 0.237 (0.93) | 0.274 (0.81) | 0.328 (0.67) | 0.122 (**1.81**) | 0.138 (1.60) |
| cav2d-rect-pert-1e5 | 99459 | AMD | 0.033 | 0.032 (1.04) | 0.026 (**1.26**) | 0.027 (1.25) | 0.045 (0.73) | 0.056 (0.59) | 0.028 (1.16) | 0.034 (0.97) |
| syn-mem125-46 | 97336 | SCOTCH | 2.295 | 2.314 (0.99) | 2.395 (0.96) | 2.524 (0.91) | 1.865 (1.23) | 1.954 (1.17) | 0.990 (**2.32**) | 1.050 (2.19) |
| ss-tem152078 | 152078 | SCOTCH | 0.555 | 0.573 (0.97) | 0.659 (0.84) | 0.651 (0.85) | 0.719 (0.77) | 0.669 (0.83) | 0.503 (**1.10**) | 0.503 (1.10) |
| wg-square-1e6 | 998001 | AMD | 0.659 | 0.423 (1.56) | 0.433 (1.52) | 0.731 (0.90) | 0.853 (0.77) | 1.025 (0.64) | 0.320 (**2.06**) | 0.467 (1.41) |
| syn-mem27-100 | 1000000 | SCOTCH | 28.94 | - | 29.10 (0.99) | - | 21.05 (1.37) | - | 10.02 (**2.89**) | - |

The first (cold) factorization of a GPU variant costs 0.05-0.18 s more than a warm one (cuBLAS and
kernel loading; MUMPS: 0.01-0.03 s), on top of the CUDA context (0.13-0.15 s, 543 MiB of GPU
memory). On wg-square-1e5 and cav2d-rect-pert-1e5 no subtree reaches min_gpu_work: their "GPU" runs
are CPU runs in SSIDS' GPU mode (slower with the default binding, see Tuning).

**Solves (s): one column / eight columns**

| case | mumps:driver | ssids-cpu:metis | ssids-gpu:metis | tuned gpu:metis |
|---|---|---|---|---|
| syn-mem27-46 | 0.0437 / 0.0474 | 0.0545 / 0.1008 | 0.0373 / 0.0944 | 0.0253 / 0.0842 |
| yee-pec-box-1e5 | 0.0440 / 0.0492 | 0.0668 / 0.1247 | 0.0424 / 0.1200 | 0.0415 / 0.1155 |
| wg-square-1e5 | 0.0080 / 0.0161 | 0.0145 / 0.0397 | 0.0295 / 0.0404 | 0.0146 / 0.0400 |
| ss-qa8fk-qa8fm | 0.0235 / 0.0264 | 0.0299 / 0.0563 | 0.0239 / 0.0566 | 0.0169 / 0.0466 |
| cav2d-rect-pert-1e5 | 0.0058 / 0.0128 | 0.0116 / 0.0360 | 0.0229 / 0.0365 | 0.0117 / 0.0359 |
| syn-mem125-46 | 0.1089 / 0.1251 | 0.1357 / 0.2053 | 0.0936 / 0.1784 | 0.0588 / 0.1381 |
| ss-tem152078 | 0.0520 / 0.0586 | 0.0739 / 0.1380 | 0.0616 / 0.1443 | 0.0504 / 0.1467 |
| wg-square-1e6 | 0.0941 / 0.1479 | 0.1633 / 0.4342 | 0.1632 / 0.4491 | 0.1229 / 0.3938 |
| syn-mem27-100 | 0.801 / 0.933 | 1.005 / 1.674 | 0.638 / 1.234 | 0.394 / 0.896 |

MUMPS solves eight columns for 1.1-2.2x the time of one (hsolve's reference runs spend 6-11 ms per
right-hand side on the 1e5 3-D cases: it solves in blocks), SSIDS for 1.5-3.1x on the CPU and
2.3-3.3x with the tuned GPU (one right-hand side after the other in the GPU subtrees).

**Analysis (s)**, `mumps:driver` (AMD probe + SCOTCH where the cost model picks it) vs SSIDS with
METIS: syn-mem27-46 0.72 / 1.27, yee-pec-box-1e5 0.63 / 1.17, wg-square-1e5 0.049 / 0.67,
ss-qa8fk-qa8fm 0.54 / 0.83, cav2d-rect-pert-1e5 0.029 / 0.46, syn-mem125-46 1.96 / 3.02,
ss-tem152078 0.90 / 1.85, wg-square-1e6 0.54 / 8.02, syn-mem27-100 6.7 / 16.2. Given MUMPS' order,
SSIDS analyses in 0.02-0.4 s, but that order costs the MUMPS analysis. Side finding: inside the
SSIDS processes (OMP_PROC_BIND=TRUE, which SSIDS asks for) that same MUMPS analysis took 2.0-8.2 s
instead of 0.5-2.0 s on the SCOTCH cases (0.55-1.9 s in the tuned runs with OMP_PROC_BIND=FALSE):
binding pins the initial thread to one core and the ordering's threads evidently stay there. MUMPS
and a bound SSIDS would not mix well in one process.

**Accuracy**: backward errors (inf-norm, worst of the one- and eight-column solves of all runs,
no refinement) 2e-22..1.1e-11 for SSIDS and 7e-22..1.8e-11 for MUMPS; SSIDS on the CPU is about an
order of magnitude lower on the 3-D cases, its GPU variants about as MUMPS.

**Memory: host peak GiB of the backend phase (incl. matrices); GPU peak MiB incl. the 543 MiB context**

| case | mumps:driver | ssids-cpu:metis | ssids-gpu:metis (GPU) | tuned gpu:metis (GPU) |
|---|---|---|---|---|
| syn-mem27-46 | 1.09 | 1.43 | 1.30 (945) | 1.01 (1225) |
| yee-pec-box-1e5 | 1.05 | 1.58 | 1.26 (1167) | 1.41 (1183) |
| wg-square-1e5 | 0.33 | 0.30 | 0.44 (543) | 0.45 (543) |
| ss-qa8fk-qa8fm | 0.79 | 0.85 | 0.88 (745) | 0.77 (857) |
| cav2d-rect-pert-1e5 | 0.25 | 0.25 | 0.39 (543) | 0.38 (543) |
| syn-mem125-46 | 2.98 | 4.53 | 3.96 (1735) | 2.69 (2835) |
| ss-tem152078 | 1.52 | 1.85 | 2.04 (885) | 1.88 (1007) |
| wg-square-1e6 | 1.88 | 2.20 | 2.50 (543) | 2.25 (985) |
| syn-mem27-100 | 14.95 | 28.15 | 22.38 (9447) | 14.69 (15717) |

SSIDS on the CPU needs 1.0-1.9x MUMPS' host memory (28 GiB of the 32 GiB limit on syn-mem27-100);
the tuned GPU run keeps enough factors on the card to stay at MUMPS' host level, but used 15.7 of
the 16 GB there: a larger 3-D problem would not fit on one V100.

## Inertia agreement

Every factorization of every variant and repetition reported the same number of negative pivots as
`mumps:driver` and no zero pivot (`zero = n - matrix_rank` for SSIDS, INFOG(28) for MUMPS):
cav2d-rect-pert-1e5 32900, ss-qa8fk-qa8fm 79, ss-tem152078 141072, syn-mem125-46 1947,
syn-mem27-100 20003, syn-mem27-46 1950, wg-square-1e5 24659, wg-square-1e6 249011,
yee-pec-box-1e5 35478. For the synthetic cases (sigma = 0, B = I) these are the exact counts of
`meta.json`. No mismatch; SSIDS (u = 0.01) delayed 0-5 pivots per factorization. These shifts are
generic (no eigenvalue at sigma): near-singular shifts, where MUMPS' null-pivot detection (CNTL(3) =
1e-8, relative) and SSIDS' `small` = 1e-20 (absolute) would behave differently, were not tested.

## Projection: hsolve's 16-core total with SSIDS (a projection, not a measurement)

Reference: `results/cmp-p16-best.json` (hsolve with / without the inertia certificate, 16 cores on
gpu01). Each phase is scaled by the ratio measured here (`tools/bench/gpu_g1_report.py`): analysis by
analyse_X / analyse_M (`:mumps` orderings: SSIDS' analysis + MUMPS' own analysis time); the numeric
factorizations by (cold_X + (F-1) warm_X) / (cold_M + (F-1) warm_M) with the run's F; the solves by
the eight-column ratio (hsolve solves in blocks; in brackets the one-column ratio as an upper bound);
the certificate's InertiaCounter counts (syn-mem27-46, yee-pec-box-1e5: hsolve minus hsolve-nocert)
by warm_X / (0.87 warm_M), since SSIDS has no inertia-only mode; plus the CUDA context for GPU
variants. Iteration and verification time unchanged. Seconds, hsolve / hsolve-nocert:

| case | reference | ssids-cpu:metis | ssids-cpu:mumps | tuned gpu:metis | tuned gpu:mumps |
|---|---|---|---|---|---|
| syn-mem27-46 | 2.76 / 1.85 | 3.90 / 2.86 | 3.47 / 2.43 | 3.24 / 2.70 [2.68 / 2.13] | 2.93 / 2.33 [2.30 / 1.70] |
| yee-pec-box-1e5 | 3.24 / 2.31 | 6.53 / 4.55 | 5.67 / 3.92 | 5.33 / 4.16 [4.06 / 2.89] | 4.55 / 3.74 [3.03 / 2.23] |
| wg-square-1e5 | 0.59 / 0.59 | 1.52 / 1.52 | 1.05 / 1.05 | 1.67 / 1.67 [1.50] | 1.18 / 1.18 [1.04] |
| ss-qa8fk-qa8fm | 2.32 / 1.82 | 3.24 / 2.74 | 3.22 / 2.67 | 2.83 / 2.53 [2.15 / 1.87] | 2.89 / 2.55 [2.11 / 1.79] |
| cav2d-rect-pert-1e5 | 0.34 / 0.28 | 0.96 / 0.92 | 0.56 / 0.51 | 1.09 / 1.04 [0.99 / 0.94] | 0.72 / 0.66 [0.62 / 0.55] |

Every projected total is above hsolve's (ratios 0.27-0.94x; SSIDS GPU with its defaults lies
between the CPU and the tuned columns, see the appendix). The best variant projects 6-41% slower on
the 3-D reference cases and 1.6-1.8x slower on the 2-D ones. Only the one-column bound, which assumes
SSIDS' block solves had MUMPS' per-column efficiency, reaches 1.20x / 1.09x (syn-mem27-46, tuned GPU
with MUMPS' order), 1.07x (yee-pec-box-1e5) and 1.10x (ss-qa8fk-qa8fm). syn-mem125-46,
ss-tem152078, wg-square-1e6 and syn-mem27-100 have no hsolve reference run, hence no projection.

## Conclusion

- **G2 is not worth it.** The bar - factorization >= 2x faster on the 3-D cases with identical
  inertia - is met for the inertia (always identical) but not for the speed: with one V100 SSIDS
  factorizes 0.9-1.85x as fast as MUMPS on the 1e5-unknown 3-D cases, and that only after working
  around SSIDS' thread binding (defaults: 0.7-1.0x); >= 2x only from ~5e11 flops per factorization
  (syn-mem125-46 2.3x, syn-mem27-100 2.9x). Its block solves are ~2x slower, its analysis slower and
  the CUDA context adds 0.13-0.15 s, so on the competition's cases the projected solver time grows
  by 6% to 3.7x; the one problem class where the GPU clearly wins (1e6 3-D) already fills the 16 GB
  card. The reasons are structural in SSIDS: only leaf subtrees go to the GPU, the top separators
  (the bulk of the 3-D work) are factorized on the CPU after the GPU phase, and GPU solves treat one
  right-hand side at a time.
- **SSIDS CPU-only does not beat MUMPS anywhere end to end.** Its factorization is faster only on
  2-D problems where it orders with METIS and the driver picks AMD (wg-square-1e5 1.54x, cav2d
  1.26x, wg-square-1e6 1.52x; MUMPS with METIS gains the same, but METIS' analysis costs more than
  the saving, which is why the driver picks AMD); its solves are 1.6-2.9x slower per block and it
  needs up to 1.9x the host memory.
- What could change the verdict is a GPU path that also factorizes the top separators on the GPU and
  solves blocks of right-hand sides, for problems of >= ~5e11 flops - a different experiment than
  integrating SSIDS.

## Not measured / limitations

- Complex Hermitian cases: SSIDS is real symmetric only; the order-2n real embedding (which hsolve
  feeds MUMPS) was not tried; all nine cases are real.
- hsolve itself was not run with SSIDS: the totals above are projections from phase ratios (same
  Krylov iteration counts assumed). Four cases have no hsolve reference breakdown, so no projection.
- Inertia-only counts: SSIDS has no discard-factors mode; a count was assumed to cost a full
  factorization (not measured separately).
- Near-singular shifts (SSIDS' zero-pivot handling vs MUMPS' null-pivot detection), scalings and
  thresholds other than Ruiz with u = 0.01, out-of-core, more than one GPU.
- syn-mem27-100: only mumps:driver, ssids-cpu:metis and ssids-gpu:metis (default and tuned).
- The tuning (gpu_perf_coeff, binding) was chosen on two cases with one repetition and applied
  unchanged to all; per-case tuning might gain somewhat more.

## Appendix: full table (`python3 tools/bench/gpu_g1_report.py results/gpu_g1.jsonl`)

Medians over the repetitions; speedups vs `mumps:driver` of the same case; max resid = worst backward error of all runs; bracketed arguments = the tuned runs (and the syn-mem27-100 settings). The projection rows use the eight-column solve ratio.

| case | variant | ordering | analyse s | factor cold s | factor warm s | solve-1 s | solve-p s | factor speedup | solve-1 speedup | solve-p speedup | neg | inertia | max resid | host peak GiB | GPU peak MiB | runs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| syn-mem27-46 | mumps:driver | driver (scotch) | 0.722 | 0.519 | 0.494 | 0.0437 | 0.0474 | 1.00x | 1.00x | 1.00x | 1950 | ok | 2.8e-12 | 1.09 | - | 3/3 |
| syn-mem27-46 | mumps:metis | metis (metis) | 1.082 | 0.482 | 0.467 | 0.0439 | 0.0491 | 1.06x | 0.99x | 0.96x | 1950 | ok | 2.6e-12 | 0.86 | - | 3/3 |
| syn-mem27-46 | ssids-cpu:metis | metis | 1.272 | 0.493 | 0.499 | 0.0545 | 0.1008 | 0.99x | 0.80x | 0.47x | 1950 | ok | 1.9e-13 | 1.43 | - | 3/3 |
| syn-mem27-46 | ssids-cpu:mumps | mumps (scotch) | 0.061 | 0.498 | 0.494 | 0.0556 | 0.1020 | 1.00x | 0.79x | 0.46x | 1950 | ok | 6.6e-13 | 1.67 | - | 3/3 |
| syn-mem27-46 | ssids-gpu:metis | metis | 1.285 | 0.519 | 0.492 | 0.0373 | 0.0944 | 1.00x | 1.17x | 0.50x | 1950 | ok | 5.4e-13 | 1.30 | 945 | 3/3 |
| syn-mem27-46 | ssids-gpu:metis [--gpu-perf-coeff 4 OMP_PROC_BIND=FALSE] | metis | 1.279 | 0.355 | 0.267 | 0.0253 | 0.0842 | 1.85x | 1.73x | 0.56x | 1950 | ok | 7.2e-13 | 1.01 | 1225 | 3/3 |
| syn-mem27-46 | ssids-gpu:mumps | mumps (scotch) | 0.067 | 0.545 | 0.510 | 0.0396 | 0.1023 | 0.97x | 1.10x | 0.46x | 1950 | ok | 3.5e-13 | 1.58 | 935 | 3/3 |
| syn-mem27-46 | ssids-gpu:mumps [--gpu-perf-coeff 4 OMP_PROC_BIND=FALSE] | mumps (scotch) | 0.066 | 0.374 | 0.285 | 0.0255 | 0.0911 | 1.74x | 1.71x | 0.52x | 1950 | ok | 1.6e-12 | 1.19 | 1231 | 3/3 |
| yee-pec-box-1e5 | mumps:driver | driver (scotch) | 0.634 | 0.507 | 0.489 | 0.0440 | 0.0492 | 1.00x | 1.00x | 1.00x | 35478 | ok | 2.6e-13 | 1.05 | - | 3/3 |
| yee-pec-box-1e5 | mumps:metis | metis (metis) | 1.227 | 0.533 | 0.523 | 0.0452 | 0.0515 | 0.93x | 0.97x | 0.96x | 35478 | ok | 6.7e-13 | 0.88 | - | 3/3 |
| yee-pec-box-1e5 | ssids-cpu:metis | metis | 1.174 | 0.898 | 0.895 | 0.0668 | 0.1247 | 0.55x | 0.66x | 0.39x | 35478 | ok | 1.7e-14 | 1.58 | - | 3/3 |
| yee-pec-box-1e5 | ssids-cpu:mumps | mumps (scotch) | 0.063 | 0.790 | 0.790 | 0.0630 | 0.1187 | 0.62x | 0.70x | 0.41x | 35478 | ok | 3.3e-14 | 1.68 | - | 3/3 |
| yee-pec-box-1e5 | ssids-gpu:metis | metis | 1.181 | 0.725 | 0.684 | 0.0424 | 0.1200 | 0.71x | 1.04x | 0.41x | 35478 | ok | 2.0e-14 | 1.26 | 1167 | 3/3 |
| yee-pec-box-1e5 | ssids-gpu:metis [--gpu-perf-coeff 4 OMP_PROC_BIND=FALSE] | metis | 1.175 | 0.534 | 0.532 | 0.0415 | 0.1155 | 0.92x | 1.06x | 0.43x | 35478 | ok | 2.0e-14 | 1.41 | 1183 | 3/3 |
| yee-pec-box-1e5 | ssids-gpu:mumps | mumps (scotch) | 0.071 | 0.657 | 0.620 | 0.0459 | 0.1363 | 0.79x | 0.96x | 0.36x | 35478 | ok | 2.2e-14 | 1.51 | 989 | 3/3 |
| yee-pec-box-1e5 | ssids-gpu:mumps [--gpu-perf-coeff 4 OMP_PROC_BIND=FALSE] | mumps (scotch) | 0.069 | 0.448 | 0.365 | 0.0340 | 0.1201 | 1.34x | 1.29x | 0.41x | 35478 | ok | 6.4e-14 | 1.21 | 1143 | 3/3 |
| wg-square-1e5 | mumps:driver | driver (amd) | 0.049 | 0.066 | 0.059 | 0.0080 | 0.0161 | 1.00x | 1.00x | 1.00x | 24659 | ok | 4.9e-15 | 0.33 | - | 3/3 |
| wg-square-1e5 | mumps:metis | metis (metis) | 0.689 | 0.057 | 0.048 | 0.0072 | 0.0138 | 1.23x | 1.12x | 1.17x | 24659 | ok | 1.1e-15 | 0.32 | - | 3/3 |
| wg-square-1e5 | ssids-cpu:metis | metis | 0.672 | 0.038 | 0.038 | 0.0145 | 0.0397 | 1.54x | 0.56x | 0.41x | 24659 | ok | 2.5e-17 | 0.30 | - | 3/3 |
| wg-square-1e5 | ssids-cpu:mumps | mumps (amd) | 0.039 | 0.059 | 0.058 | 0.0171 | 0.0440 | 1.03x | 0.47x | 0.37x | 24659 | ok | 2.7e-17 | 0.35 | - | 3/3 |
| wg-square-1e5 | ssids-gpu:metis | metis | 0.673 | 0.080 | 0.081 | 0.0295 | 0.0404 | 0.73x | 0.27x | 0.40x | 24659 | ok | 2.5e-17 | 0.44 | 543 | 3/3 |
| wg-square-1e5 | ssids-gpu:metis [--gpu-perf-coeff 4 OMP_PROC_BIND=FALSE] | metis | 0.670 | 0.045 | 0.041 | 0.0146 | 0.0400 | 1.45x | 0.55x | 0.40x | 24659 | ok | 2.5e-17 | 0.45 | 543 | 3/3 |
| wg-square-1e5 | ssids-gpu:mumps | mumps (amd) | 0.040 | 0.154 | 0.132 | 0.0309 | 0.0443 | 0.45x | 0.26x | 0.36x | 24659 | ok | 2.7e-17 | 0.49 | 543 | 3/3 |
| wg-square-1e5 | ssids-gpu:mumps [--gpu-perf-coeff 4 OMP_PROC_BIND=FALSE] | mumps (amd) | 0.040 | 0.064 | 0.059 | 0.0171 | 0.0439 | 1.01x | 0.47x | 0.37x | 24659 | ok | 2.7e-17 | 0.49 | 543 | 3/3 |
| ss-qa8fk-qa8fm | mumps:driver | driver (scotch) | 0.537 | 0.246 | 0.221 | 0.0235 | 0.0264 | 1.00x | 1.00x | 1.00x | 79 | ok | 2.2e-12 | 0.79 | - | 3/3 |
| ss-qa8fk-qa8fm | mumps:metis | metis (metis) | 0.717 | 0.221 | 0.209 | 0.0233 | 0.0280 | 1.05x | 1.01x | 0.94x | 79 | ok | 3.6e-13 | 0.55 | - | 3/3 |
| ss-qa8fk-qa8fm | ssids-cpu:metis | metis | 0.830 | 0.212 | 0.211 | 0.0299 | 0.0563 | 1.05x | 0.79x | 0.47x | 79 | ok | 2.7e-14 | 0.85 | - | 3/3 |
| ss-qa8fk-qa8fm | ssids-cpu:mumps | mumps (scotch) | 0.044 | 0.234 | 0.237 | 0.0325 | 0.0601 | 0.93x | 0.72x | 0.44x | 79 | ok | 8.0e-14 | 1.01 | - | 3/3 |
| ss-qa8fk-qa8fm | ssids-gpu:metis | metis | 0.833 | 0.290 | 0.274 | 0.0239 | 0.0566 | 0.81x | 0.99x | 0.47x | 79 | ok | 5.8e-14 | 0.88 | 745 | 3/3 |
| ss-qa8fk-qa8fm | ssids-gpu:metis [--gpu-perf-coeff 4 OMP_PROC_BIND=FALSE] | metis | 0.829 | 0.177 | 0.122 | 0.0169 | 0.0466 | 1.81x | 1.39x | 0.57x | 79 | ok | 9.2e-14 | 0.77 | 857 | 3/3 |
| ss-qa8fk-qa8fm | ssids-gpu:mumps | mumps (scotch) | 0.041 | 0.348 | 0.328 | 0.0335 | 0.0609 | 0.67x | 0.70x | 0.43x | 79 | ok | 5.3e-13 | 1.30 | 543 | 3/3 |
| ss-qa8fk-qa8fm | ssids-gpu:mumps [--gpu-perf-coeff 4 OMP_PROC_BIND=FALSE] | mumps (scotch) | 0.048 | 0.221 | 0.138 | 0.0198 | 0.0537 | 1.60x | 1.19x | 0.49x | 79 | ok | 9.5e-14 | 1.08 | 843 | 3/3 |
| cav2d-rect-pert-1e5 | mumps:driver | driver (amd) | 0.029 | 0.038 | 0.033 | 0.0058 | 0.0128 | 1.00x | 1.00x | 1.00x | 32900 | ok | 6.1e-17 | 0.25 | - | 3/3 |
| cav2d-rect-pert-1e5 | mumps:metis | metis (metis) | 0.456 | 0.037 | 0.032 | 0.0063 | 0.0123 | 1.04x | 0.92x | 1.05x | 32900 | ok | 9.4e-17 | 0.28 | - | 3/3 |
| cav2d-rect-pert-1e5 | ssids-cpu:metis | metis | 0.458 | 0.027 | 0.026 | 0.0116 | 0.0360 | 1.26x | 0.50x | 0.36x | 32900 | ok | 6.5e-17 | 0.25 | - | 3/3 |
| cav2d-rect-pert-1e5 | ssids-cpu:mumps | mumps (amd) | 0.017 | 0.026 | 0.027 | 0.0119 | 0.0365 | 1.25x | 0.49x | 0.35x | 32900 | ok | 8.3e-17 | 0.26 | - | 3/3 |
| cav2d-rect-pert-1e5 | ssids-gpu:metis | metis | 0.445 | 0.043 | 0.045 | 0.0229 | 0.0365 | 0.73x | 0.25x | 0.35x | 32900 | ok | 6.5e-17 | 0.39 | 543 | 3/3 |
| cav2d-rect-pert-1e5 | ssids-gpu:metis [--gpu-perf-coeff 4 OMP_PROC_BIND=FALSE] | metis | 0.443 | 0.026 | 0.028 | 0.0117 | 0.0359 | 1.16x | 0.50x | 0.36x | 32900 | ok | 6.5e-17 | 0.38 | 543 | 3/3 |
| cav2d-rect-pert-1e5 | ssids-gpu:mumps | mumps (amd) | 0.019 | 0.045 | 0.056 | 0.0240 | 0.0364 | 0.59x | 0.24x | 0.35x | 32900 | ok | 8.3e-17 | 0.40 | 543 | 3/3 |
| cav2d-rect-pert-1e5 | ssids-gpu:mumps [--gpu-perf-coeff 4 OMP_PROC_BIND=FALSE] | mumps (amd) | 0.019 | 0.028 | 0.034 | 0.0118 | 0.0366 | 0.97x | 0.49x | 0.35x | 32900 | ok | 8.3e-17 | 0.41 | 543 | 3/3 |
| syn-mem125-46 | mumps:driver | driver (scotch) | 1.958 | 2.324 | 2.295 | 0.1089 | 0.1251 | 1.00x | 1.00x | 1.00x | 1947 | ok | 3.0e-12 | 2.98 | - | 3/3 |
| syn-mem125-46 | mumps:metis | metis (metis) | 2.152 | 2.308 | 2.314 | 0.1023 | 0.1226 | 0.99x | 1.06x | 1.02x | 1947 | ok | 2.7e-12 | 2.52 | - | 3/3 |
| syn-mem125-46 | ssids-cpu:metis | metis | 3.021 | 2.394 | 2.395 | 0.1357 | 0.2053 | 0.96x | 0.80x | 0.61x | 1947 | ok | 3.7e-13 | 4.53 | - | 3/3 |
| syn-mem125-46 | ssids-cpu:mumps | mumps (scotch) | 0.211 | 2.540 | 2.524 | 0.1423 | 0.2151 | 0.91x | 0.77x | 0.58x | 1947 | ok | 2.4e-12 | 5.17 | - | 3/3 |
| syn-mem125-46 | ssids-gpu:metis | metis | 3.040 | 1.859 | 1.865 | 0.0936 | 0.1784 | 1.23x | 1.16x | 0.70x | 1947 | ok | 2.4e-13 | 3.96 | 1735 | 3/3 |
| syn-mem125-46 | ssids-gpu:metis [--gpu-perf-coeff 4 OMP_PROC_BIND=FALSE] | metis | 3.064 | 1.157 | 0.990 | 0.0588 | 0.1381 | 2.32x | 1.85x | 0.91x | 1947 | ok | 3.5e-13 | 2.69 | 2835 | 3/3 |
| syn-mem125-46 | ssids-gpu:mumps | mumps (scotch) | 0.223 | 2.033 | 1.954 | 0.1022 | 0.2088 | 1.17x | 1.07x | 0.60x | 1947 | ok | 1.7e-12 | 4.62 | 1873 | 3/3 |
| syn-mem125-46 | ssids-gpu:mumps [--gpu-perf-coeff 4 OMP_PROC_BIND=FALSE] | mumps (scotch) | 0.222 | 1.230 | 1.050 | 0.0636 | 0.1511 | 2.19x | 1.71x | 0.83x | 1947 | ok | 4.0e-12 | 3.20 | 2871 | 3/3 |
| ss-tem152078 | mumps:driver | driver (scotch) | 0.895 | 0.582 | 0.555 | 0.0520 | 0.0586 | 1.00x | 1.00x | 1.00x | 141072 | ok | 2.1e-21 | 1.52 | - | 3/3 |
| ss-tem152078 | mumps:metis | metis (metis) | 1.779 | 0.597 | 0.573 | 0.0540 | 0.0596 | 0.97x | 0.96x | 0.98x | 141072 | ok | 9.5e-22 | 1.30 | - | 3/3 |
| ss-tem152078 | ssids-cpu:metis | metis | 1.847 | 0.657 | 0.659 | 0.0739 | 0.1380 | 0.84x | 0.70x | 0.42x | 141072 | ok | 1.9e-22 | 1.85 | - | 3/3 |
| ss-tem152078 | ssids-cpu:mumps | mumps (scotch) | 0.171 | 0.660 | 0.651 | 0.0750 | 0.1394 | 0.85x | 0.69x | 0.42x | 141072 | ok | 3.7e-22 | 1.98 | - | 3/3 |
| ss-tem152078 | ssids-gpu:metis | metis | 1.855 | 0.748 | 0.719 | 0.0616 | 0.1443 | 0.77x | 0.84x | 0.41x | 141072 | ok | 2.9e-22 | 2.04 | 885 | 3/3 |
| ss-tem152078 | ssids-gpu:metis [--gpu-perf-coeff 4 OMP_PROC_BIND=FALSE] | metis | 1.852 | 0.633 | 0.503 | 0.0504 | 0.1467 | 1.10x | 1.03x | 0.40x | 141072 | ok | 6.6e-22 | 1.88 | 1007 | 3/3 |
| ss-tem152078 | ssids-gpu:mumps | mumps (scotch) | 0.175 | 0.637 | 0.669 | 0.0494 | 0.1257 | 0.83x | 1.05x | 0.47x | 141072 | ok | 1.5e-21 | 1.78 | 1079 | 3/3 |
| ss-tem152078 | ssids-gpu:mumps [--gpu-perf-coeff 4 OMP_PROC_BIND=FALSE] | mumps (scotch) | 0.175 | 0.562 | 0.503 | 0.0526 | 0.1344 | 1.10x | 0.99x | 0.44x | 141072 | ok | 1.8e-21 | 2.04 | 1003 | 3/3 |
| wg-square-1e6 | mumps:driver | driver (amd) | 0.537 | 0.666 | 0.659 | 0.0941 | 0.1479 | 1.00x | 1.00x | 1.00x | 249011 | ok | 7.1e-16 | 1.88 | - | 3/3 |
| wg-square-1e6 | mumps:metis | metis (metis) | 8.237 | 0.472 | 0.423 | 0.0785 | 0.1336 | 1.56x | 1.20x | 1.11x | 249011 | ok | 6.4e-16 | 1.55 | - | 3/3 |
| wg-square-1e6 | ssids-cpu:metis | metis | 8.023 | 0.453 | 0.433 | 0.1633 | 0.4342 | 1.52x | 0.58x | 0.34x | 249011 | ok | 1.0e-17 | 2.20 | - | 3/3 |
| wg-square-1e6 | ssids-cpu:mumps | mumps (amd) | 0.396 | 0.722 | 0.731 | 0.2003 | 0.4905 | 0.90x | 0.47x | 0.30x | 249011 | ok | 3.5e-17 | 2.66 | - | 3/3 |
| wg-square-1e6 | ssids-gpu:metis | metis | 8.060 | 0.949 | 0.853 | 0.1632 | 0.4491 | 0.77x | 0.58x | 0.33x | 249011 | ok | 9.4e-18 | 2.50 | 543 | 3/3 |
| wg-square-1e6 | ssids-gpu:metis [--gpu-perf-coeff 4 OMP_PROC_BIND=FALSE] | metis | 8.059 | 0.364 | 0.320 | 0.1229 | 0.3938 | 2.06x | 0.77x | 0.38x | 249011 | ok | 8.6e-18 | 2.25 | 985 | 3/3 |
| wg-square-1e6 | ssids-gpu:mumps | mumps (amd) | 0.409 | 1.024 | 1.025 | 0.1401 | 0.4294 | 0.64x | 0.67x | 0.34x | 249011 | ok | 5.8e-17 | 2.58 | 1275 | 3/3 |
| wg-square-1e6 | ssids-gpu:mumps [--gpu-perf-coeff 4 OMP_PROC_BIND=FALSE] | mumps (amd) | 0.411 | 0.530 | 0.467 | 0.1149 | 0.4053 | 1.41x | 0.82x | 0.36x | 249011 | ok | 4.0e-17 | 2.33 | 1423 | 3/3 |
| syn-mem27-100 | mumps:driver [--memory-budget-gb 30 --solves 5 --block-solves 3] | driver (scotch) | 6.714 | 29.138 | 28.937 | 0.8013 | 0.9332 | 1.00x | 1.00x | 1.00x | 20003 | ok | 1.8e-11 | 14.95 | - | 3/3 |
| syn-mem27-100 | ssids-cpu:metis [--memory-budget-gb 30 --solves 5 --block-solves 3] | metis | 16.138 | 29.086 | 29.102 | 1.0047 | 1.6742 | 0.99x | 0.80x | 0.56x | 20003 | ok | 9.0e-13 | 28.15 | - | 3/3 |
| syn-mem27-100 | ssids-gpu:metis [--memory-budget-gb 30 --solves 5 --block-solves 3 --gpu-perf-coeff 4 OMP_PROC_BIND=FALSE] | metis | 16.270 | 10.031 | 10.023 | 0.3943 | 0.8955 | 2.89x | 2.03x | 1.04x | 20003 | ok | 1.1e-11 | 14.69 | 15717 | 3/3 |
| syn-mem27-100 | ssids-gpu:metis [--memory-budget-gb 30 --solves 5 --block-solves 3] | metis | 16.231 | 20.959 | 21.051 | 0.6378 | 1.2335 | 1.37x | 1.26x | 0.76x | 20003 | ok | 2.2e-12 | 22.38 | 9447 | 3/3 |

CUDA context creation (cuda_init_s, medians of 34 case/variant groups): 0.127..0.146 s; first-factorization extra (cold - warm): -0.092..0.180 s; GPU memory after the context: 543 MiB

Projection of hsolve's 16-core solver time (s) with the measured ratios (see the docstring):

| case | hsolve ref | nocert ref | variant | analysis | numeric (F) | solves | certificate | projected hsolve | projected nocert | vs ref |
|---|---|---|---|---|---|---|---|---|---|---|
| syn-mem27-46 | 2.76 | 1.85 | ssids-cpu:metis | 1.13 | 0.48 (1) | 1.00 | 1.06 | 3.90 | 2.86 | 0.71x / 0.65x |
| syn-mem27-46 | 2.76 | 1.85 | ssids-cpu:mumps | 0.69 | 0.49 (1) | 1.01 | 1.04 | 3.47 | 2.43 | 0.80x / 0.76x |
| syn-mem27-46 | 2.76 | 1.85 | ssids-gpu:metis | 1.14 | 0.51 (1) | 0.93 | 1.04 | 4.00 | 2.98 | 0.69x / 0.62x |
| syn-mem27-46 | 2.76 | 1.85 | ssids-gpu:metis [--gpu-perf-coeff 4 OMP_PROC_BIND=FALSE] | 1.13 | 0.35 (1) | 0.83 | 0.56 | 3.24 | 2.70 | 0.85x / 0.69x |
| syn-mem27-46 | 2.76 | 1.85 | ssids-gpu:mumps | 0.70 | 0.53 (1) | 1.01 | 1.08 | 3.68 | 2.61 | 0.75x / 0.71x |
| syn-mem27-46 | 2.76 | 1.85 | ssids-gpu:mumps [--gpu-perf-coeff 4 OMP_PROC_BIND=FALSE] | 0.70 | 0.37 (1) | 0.90 | 0.60 | 2.93 | 2.33 | 0.94x / 0.80x |
| yee-pec-box-1e5 | 3.24 | 2.31 | ssids-cpu:metis | 1.02 | 0.91 (1) | 2.31 | 1.97 | 6.53 | 4.55 | 0.50x / 0.51x |
| yee-pec-box-1e5 | 3.24 | 2.31 | ssids-cpu:mumps | 0.60 | 0.80 (1) | 2.20 | 1.74 | 5.67 | 3.92 | 0.57x / 0.59x |
| yee-pec-box-1e5 | 3.24 | 2.31 | ssids-gpu:metis | 1.02 | 0.73 (1) | 2.22 | 1.50 | 5.95 | 4.44 | 0.54x / 0.52x |
| yee-pec-box-1e5 | 3.24 | 2.31 | ssids-gpu:metis [--gpu-perf-coeff 4 OMP_PROC_BIND=FALSE] | 1.02 | 0.54 (1) | 2.14 | 1.17 | 5.33 | 4.16 | 0.61x / 0.55x |
| yee-pec-box-1e5 | 3.24 | 2.31 | ssids-gpu:mumps | 0.61 | 0.67 (1) | 2.52 | 1.36 | 5.62 | 4.25 | 0.58x / 0.54x |
| yee-pec-box-1e5 | 3.24 | 2.31 | ssids-gpu:mumps [--gpu-perf-coeff 4 OMP_PROC_BIND=FALSE] | 0.61 | 0.45 (1) | 2.22 | 0.80 | 4.55 | 3.74 | 0.71x / 0.62x |
| wg-square-1e5 | 0.59 | 0.59 | ssids-cpu:metis | 0.67 | 0.08 (2) | 0.60 | 2.4e-04 | 1.52 | 1.52 | 0.39x / 0.39x |
| wg-square-1e5 | 0.59 | 0.59 | ssids-cpu:mumps | 0.09 | 0.12 (2) | 0.67 | 3.6e-04 | 1.05 | 1.05 | 0.57x / 0.57x |
| wg-square-1e5 | 0.59 | 0.59 | ssids-gpu:metis | 0.67 | 0.16 (2) | 0.61 | 5.0e-04 | 1.76 | 1.76 | 0.34x / 0.34x |
| wg-square-1e5 | 0.59 | 0.59 | ssids-gpu:metis [--gpu-perf-coeff 4 OMP_PROC_BIND=FALSE] | 0.67 | 0.09 (2) | 0.61 | 2.5e-04 | 1.67 | 1.67 | 0.36x / 0.36x |
| wg-square-1e5 | 0.59 | 0.59 | ssids-gpu:mumps | 0.09 | 0.29 (2) | 0.67 | 8.2e-04 | 1.35 | 1.35 | 0.44x / 0.44x |
| wg-square-1e5 | 0.59 | 0.59 | ssids-gpu:mumps [--gpu-perf-coeff 4 OMP_PROC_BIND=FALSE] | 0.09 | 0.12 (2) | 0.67 | 3.6e-04 | 1.18 | 1.18 | 0.50x / 0.50x |
| ss-qa8fk-qa8fm | 2.32 | 1.82 | ssids-cpu:metis | 0.72 | 0.85 (4) | 1.38 | 0.00 | 3.24 | 2.74 | 0.72x / 0.66x |
| ss-qa8fk-qa8fm | 2.32 | 1.82 | ssids-cpu:mumps | 0.50 | 0.95 (4) | 1.48 | 0.00 | 3.22 | 2.67 | 0.72x / 0.68x |
| ss-qa8fk-qa8fm | 2.32 | 1.82 | ssids-gpu:metis | 0.72 | 1.12 (4) | 1.39 | 0.00 | 3.66 | 3.03 | 0.63x / 0.60x |
| ss-qa8fk-qa8fm | 2.32 | 1.82 | ssids-gpu:metis [--gpu-perf-coeff 4 OMP_PROC_BIND=FALSE] | 0.72 | 0.55 (4) | 1.14 | 0.00 | 2.83 | 2.53 | 0.82x / 0.72x |
| ss-qa8fk-qa8fm | 2.32 | 1.82 | ssids-gpu:mumps | 0.50 | 1.34 (4) | 1.49 | 0.00 | 3.75 | 3.01 | 0.62x / 0.61x |
| ss-qa8fk-qa8fm | 2.32 | 1.82 | ssids-gpu:mumps [--gpu-perf-coeff 4 OMP_PROC_BIND=FALSE] | 0.51 | 0.64 (4) | 1.32 | 0.00 | 2.89 | 2.55 | 0.80x / 0.72x |
| cav2d-rect-pert-1e5 | 0.34 | 0.28 | ssids-cpu:metis | 0.45 | 0.11 (4) | 0.36 | 0.00 | 0.96 | 0.92 | 0.35x / 0.30x |
| cav2d-rect-pert-1e5 | 0.34 | 0.28 | ssids-cpu:mumps | 0.04 | 0.11 (4) | 0.36 | 0.00 | 0.56 | 0.51 | 0.61x / 0.54x |
| cav2d-rect-pert-1e5 | 0.34 | 0.28 | ssids-gpu:metis | 0.44 | 0.18 (4) | 0.36 | 0.00 | 1.16 | 1.08 | 0.29x / 0.26x |
| cav2d-rect-pert-1e5 | 0.34 | 0.28 | ssids-gpu:metis [--gpu-perf-coeff 4 OMP_PROC_BIND=FALSE] | 0.44 | 0.11 (4) | 0.36 | 0.00 | 1.09 | 1.04 | 0.31x / 0.27x |
| cav2d-rect-pert-1e5 | 0.34 | 0.28 | ssids-gpu:mumps | 0.05 | 0.21 (4) | 0.36 | 0.00 | 0.80 | 0.69 | 0.43x / 0.40x |
| cav2d-rect-pert-1e5 | 0.34 | 0.28 | ssids-gpu:mumps [--gpu-perf-coeff 4 OMP_PROC_BIND=FALSE] | 0.05 | 0.13 (4) | 0.36 | 0.00 | 0.72 | 0.66 | 0.47x / 0.42x |
