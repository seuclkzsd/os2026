# P10 "gpu": offloading the factorizations' dense updates to one GPU

P10 task "gpu" (branch `p10gpu` from p9-merge 90e7483). The statement (item f, "并行方案") asks for
multithreading, MPI **and GPU** acceleration with the division of labour and measured speedups, and a
reproducible degrade mode without a GPU - the judging machine is CPU-only. This report covers the GPU
part: what can go to one V100, how it is done, what it gains, what it costs, and what it does not do.

**Answers.**

* **What can go to the GPU, and how.** Only the dense part of MUMPS' numeric factorizations is worth
  moving: at n = 1e6 a nearest run is ~75% factorizations (the main one and the exact certificate's
  two inertia counts), the solves are bound by host memory bandwidth and the factors live in host
  memory. hsolve's BLAS hook now has a GPU route (`src/gpu`, docs/DESIGN.md §3.9, docs/BLAS.md §10):
  inside a factorization, dgemm calls made outside MUMPS' OpenMP-parallel L0 subtrees go to one GPU
  through FP64 cuBLAS when a host-traffic model predicts >= 1.25x - in practice MUMPS' contribution-
  block updates (k = the front's pivots >= ~1000), **43-47% of a 1e6 factorization's flops but only
  22% of its CPU time**. The CUDA runtime and cuBLAS are opened at run time (dlopen): no CUDA at
  build time, no dependency of the binary, the same build on the CPU-only judging machine.
* **Per call 1.7-5.0x, bound by the host's memory bandwidth, not by PCIe or the GPU.** The engine
  stages each tile through pinned memory (non-temporal stores), uploads it, runs cublasDgemm and adds
  the result into C on the host; every staged byte is read, written and read again by the copy engine,
  so a call costs ~(24 (m + n) k + 32 m n) bytes / 30 GB/s - the socket's bandwidth (section 2). The
  V100 (5.6-7.0 TFlop/s FP64) is busy ~20% of an offloaded call. A hybrid split of each call between CPU
  and GPU is slower than GPU-only (both need the same bandwidth), and so are the other variants tried
  (section 8).
* **Factorizations at n = 1e6, equal pivot order, 5 interleaved runs: 1.055-1.099x each; the three
  of a run 1.092x (syn-mem27-100, 67.5 -> 61.8 s) and 1.068x (yee-pec-box-1e6, 55.6 -> 52.0 s)**
  with MUMPS' default block sizes; with the P9 block sizes (2000-row CB blocks: twice the work per byte
  moved) **1.066-1.118x each, 1.110x / 1.082x per run** (63.7 -> 57.4 s, 53.0 -> 49.0 s); with 8 CPU
  threads 1.151x (one run).
* **Whole runs at n = 1e6** (certificate exact, 5 interleaved runs, every run valid, `--gpu auto`):
  **1.069x (syn-mem27-100, 88.9 -> 83.1 s) and 1.058x (yee-pec-box-1e6, 73.8 -> 69.7 s)** - the
  factorizations' saving; the iteration (memory-bound block solves) is unchanged.
* **Below ~5e5 unknowns the GPU loses** (`on`: 0.72-0.97x on eight 3-D cases with n = 6.6e4-2.6e5 -
  the 0.25-0.36 s set-up and nothing worth offloading; break-even 1.008x at syn-mem27-80, n = 5.1e5,
  1.8e12 flops per factorization). Hence `auto` offloads only factorizations whose MUMPS estimate is
  >= 3e12 flops (the 1e6 cases: 5.4-6.7e12) and is exactly the CPU path below - measured: 0.997-1.007x
  of off on five cases, no CUDA library loaded.
* **Exactness: the counts are the CPU path's.** cuBLAS FP64 is IEEE double like the CPU BLAS; every
  element of C is one product over the full K plus one double addition, i.e. rounding-level
  differences (<= 7e-15 relative per call). The inertia of every factorization measured equals the
  GPU-off run's and the complete references' counts, every whole run's certificate count equals the
  CPU run's, and the eigenvalues agree to 3e-15 relative. GPU-only calls are bitwise reproducible.
* **Default off, degrade mode, cost.** The build's default mode is `off` - exactly today's CPU path,
  no CUDA library touched (our measurement nodes' GPUs belong to other jobs); `--gpu auto|on` /
  `HS_GPU` turn it on. Without a GPU, a CUDA library or enough free device memory the run is the CPU
  path and the report says why; a CUDA error mid-run finishes on the CPU. Cost when used: 0.25-0.39 s
  set-up per process, +0.4-0.5 GiB peak host memory (CUDA context, 224 MB pinned staging), 1 GiB +
  context on the GPU. Tests cover both configurations (section 7).
* **Division of labour** (for the "并行方案" section): section 6.

Raw runs: `results/p10_gpu.json` (every factorization, whole run and micro-benchmark line of this
report, keys `factorizations`, `whole_runs_1e6`, `whole_runs_small`, `size_screen`, `micro`; the tables
below are their medians).

## 1. Setup

* **Where.** gpu02 (2x Xeon Silver 4314, 2 x 16 cores, AVX-512, 251 GB, CentOS 7, glibc 2.17), the
  measurement slot: cores 16-31 = NUMA node 1, 32 GiB, one run at a time (`guard.py --exclusive`),
  16 threads; **GPU 7 only** - a Tesla V100-PCIE-16GB (80 SMs, compute capability 7.0, driver 535.54.03,
  CUDA 12.2 runtime and cuBLAS 12.2.1 from `/usr/local/cuda-12.2`), on the same socket as the slot (PCIe
  3 x16 behind a switch shared with GPUs 4-6, which belong to other users' jobs and were idle - 0%
  utilization - during the measurements). Every GPU run went through `tools/bench/gpu_slot.sh 7 ...`:
  nvidia-smi showed no process and <= 64 MiB used on GPU 7 before the run, the run saw only that GPU
  (`CUDA_VISIBLE_DEVICES=7`), and the slot was held with `guard.py --exclusive`. Builds and tests in
  the lane (cores 8-15). 2026-10-14.
* **Software.** The project environment (conda-forge MUMPS 5.8.2 sequential + OpenMP, OpenBLAS 0.3.34,
  gcc 13.4); hsolve at ce76a37 (built once in its own work directory, `HS_CLUSTER_NAME=p10gpu-meas`,
  for every timed run; the later commits of the branch change no timed code path), default build: BLAS hook linked into hsolve with route `all`, GPU module built
  in, GPU mode off unless a run sets `HS_GPU` / `--gpu`. MUMPS' default block sizes unless a table says
  `p9` (`HS_MUMPS_BLOCKING=p9`: KEEP(420) = 256, KEEP(57/58) = 4000/2000, results/p9_factspeed.md;
  without the L0 patch, which the conda MUMPS ignores).
* **Cases.** The two 3-D 1e6 cases in core (`$HOME/p7_p7-membudget/data`): syn-mem27-100 (HEP, n = 1e6)
  and yee-pec-box-1e6 (GHEP, n = 1 027 733); for the size dependence syn-mem27-64 / -80 (2.6e5 / 5.1e5),
  and the 1e5 3-D cases syn-mem27-46, syn-mem125-46, yee-pec-box-1e5, ss-qa8fk-qa8fm, ss-2cubes_sphere,
  ss-tem152078, yee-pec-cube-32.

## 2. The machine and the engine (`bench_gpu_dgemm`)

`bench_gpu_dgemm` (bench/gpu_dgemm.cpp; tables by `tools/bench/gpu_dgemm_table.py`), in the slot with
GPU 7, from the measurement build.

**2.1 Transfers** (`copy`, best of 5, GB/s):

| MB | memcpy pageable -> pinned (1 thread) | H2D pinned | D2H pinned | both at once, each | H2D pageable | D2H pageable | H2D pageable, strided (4000 of 15800 rows) | H2D registered | register + unregister (address range) |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 10.4 | 12.7 | 13.1 | 11.0 | 11.7 | 8.5 | 10.3 | 12.7 | 11.8 |
| 64 | 8.7 | 12.8 | 13.2 | 10.8 | 9.7 | 8.8 | 11.7 | 12.7 | 14.3 |
| 256 | 8.7 | 12.8 | 13.2 | 10.8 | 9.5 | 8.9 | 9.0 | 12.8 | 15.2 |

PCIe 3 x16 gives 12.8 GB/s per direction from pinned memory (10.8-11 GB/s each when both run at once);
copying pageable memory into a pinned buffer with one thread runs at 8.7-10.4 GB/s, and the driver's
own pageable path (synchronous staging) at 8.5-11.7 GB/s (9-11.7 strided as in a front).
`cudaHostRegister` + unregister costs 12-15 GB/s of address range. Engine set-up (`info`: library load,
context, two cuBLAS handles, 1 GiB device and 224 MB pinned buffers, threads): 0.26 s.

**2.2 cuBLAS FP64** (`devgemm`, operands resident on the device, median of 5):

| m x n x k | ms | GFlop/s |
|---|---|---|
| 1000 x 9000 x 4000 | 11.92 | 6042 |
| 1000 x 4000 x 2048 | 2.79 | 5881 |
| 2000 x 8000 x 4000 | 18.78 | 6815 |
| 1000 x 9000 x 512 | 1.54 | 6003 |
| 250 x 1000 x 4000 | 0.34 | 5846 |
| 4000 x 4000 x 128 | 0.73 | 5615 |
| 4000 x 4000 x 256 | 1.41 | 5803 |
| 8192 x 8192 x 8192 | 157.93 | 6962 |

The V100 computes every one of these shapes at 5.6-7.0 TFlop/s FP64 - 9-13x the 16-core socket's
dgemm; on the device the thin shapes are no worse than the square ones. What decides is the transfer.

**2.3 The engine against the 16-core CPU BLAS** (`gemm --ld 15800`: all operands inside a 15800-row
"front" like MUMPS' CB updates; median of 3 interleaved runs; speedup over `cpu`; the result differs
from the CPU's by <= 6.2e-15 relative to max |C| in every run). Modes: `gpu` = the default engine (GPU
only, 2 packing threads, 6 slots of 16 MB, non-temporal stores, guided tiles); `gpu-p1` / `gpu-p3` =
1 / 3 packing threads; `gpu-s3` = 3 slots; `gpu-m32` = 32 MB slots; `gpu-nt0` = regular stores;
`gpu-k1024-nt0` = 1 MB chunks uploaded as packed, regular stores (for the copy engine to read from
the LLC); `hyb` / `hyb-c8` = hybrid, the CPU computing a share with 13 / 8 BLAS threads; model = the
offload model's predicted speedup of `gpu` (the engine offloads a call when it is >= 1.25):

| m x n x k | cpu | gpu | gpu-p1 | gpu-p3 | gpu-s3 | gpu-m32 | gpu-nt0 | gpu-k1024-nt0 | hyb | hyb-c8 | model (gpu) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1000 x 9000 x 4000 | 118.5 ms (607 GF/s) | 38.4 ms (**3.08x**) | 47.8 ms (2.48x) | 38.2 ms (3.11x) | 41.1 ms (2.89x) | 39.9 ms (2.97x) | 46.4 ms (2.56x) | 43.5 ms (2.73x) | 41.9 ms (2.83x) | 41.7 ms (2.84x) | 2.64x |
| 2000 x 8000 x 4000 | 245.2 ms (522 GF/s) | 48.6 ms (**5.04x**) | 50.3 ms (4.88x) | 49.7 ms (4.94x) | 51.6 ms (4.75x) | 49.5 ms (4.96x) | 55.5 ms (4.42x) | 53.0 ms (4.62x) | 62.1 ms (3.95x) | 59.3 ms (4.13x) | 3.99x |
| 1000 x 4000 x 2048 | 26.4 ms (621 GF/s) | 12.8 ms (**2.06x**) | 14.3 ms (1.84x) | 13.1 ms (2.02x) | 14.1 ms (1.87x) | 12.5 ms (2.10x) | 13.7 ms (1.93x) | 13.1 ms (2.02x) | 13.3 ms (1.98x) | 13.7 ms (1.92x) | 1.90x |
| 1000 x 2000 x 2048 | 14.3 ms (573 GF/s) | 7.8 ms (**1.83x**) | 8.1 ms (1.76x) | 7.8 ms (1.83x) | 8.3 ms (1.71x) | 7.7 ms (1.85x) | 8.1 ms (1.76x) | 7.6 ms (1.88x) | 7.8 ms (1.83x) | 8.2 ms (1.74x) | 1.59x |
| 1000 x 9000 x 1024 | 31.9 ms (577 GF/s) | 19.0 ms (**1.68x**) | 20.0 ms (1.59x) | 19.7 ms (1.62x) | 20.3 ms (1.57x) | 19.3 ms (1.66x) | 20.4 ms (1.56x) | 20.0 ms (1.60x) | 19.1 ms (1.67x) | 19.8 ms (1.61x) | 1.53x |
| 1000 x 9000 x 512 | 29.2 ms (316 GF/s) | 16.0 ms (1.83x) | 16.6 ms (1.76x) | 16.1 ms (1.81x) | 17.2 ms (1.70x) | 16.0 ms (1.82x) | 16.6 ms (1.76x) | 16.9 ms (1.73x) | 13.9 ms (2.10x) | 15.0 ms (1.94x) | 0.98x |
| 4000 x 4000 x 256 | 15.6 ms (527 GF/s) | 25.2 ms (0.62x) | 24.7 ms (0.63x) | 25.2 ms (0.62x) | 24.2 ms (0.64x) | 24.8 ms (0.63x) | 26.4 ms (0.59x) | 24.8 ms (0.63x) | 18.2 ms (0.85x) | 18.2 ms (0.86x) | 0.65x |
| 250 x 1000 x 4000 | 3.5 ms (569 GF/s) | 6.2 ms (0.57x) | 6.6 ms (0.54x) | 5.4 ms (0.66x) | 5.8 ms (0.61x) | 6.5 ms (0.54x) | 5.2 ms (0.67x) | 5.1 ms (0.69x) | 5.0 ms (0.71x) | 5.4 ms (0.66x) | 0.59x |

(1000 x 9000 x 512: the CPU BLAS ran this shape at only 316 GFlop/s in this session - 520 in the
development screens - so the GPU's 1.83x here is the CPU's bad run; the model, which assumes 640, keeps
such calls on the CPU.)

* **GPU only, 2 packing threads, 6 slots, non-temporal stores** is the fastest configuration on every
  large shape; it is what the hook uses. One packing thread is 3-20% slower, three are not faster
  (they compete for memory bandwidth); 3 slots instead of 6 cost 6-9%; 32 MB slots are equal within 4%; regular
  stores into the staging buffers cost 4-17% (each line is read for ownership first); uploading 1 MB
  chunks as they are packed with regular stores, so that the copy engine could read them from the
  LLC (DDIO), is no better (0.89-1.03x).
* **Hybrid is slower than GPU-only** on the large shapes (the CPU's share and the staging compete for
  the same host bandwidth) and only ties on the small ones - not used.
* **Bound: host memory traffic.** The model t_gpu = (24 (m + n) k + 32 m n) bytes / 30 GB/s + 1 ms
  reproduces the GPU-only times within ~10% (e.g. 1000 x 9000 x 4000: 1.25 GB of host traffic in
  ~38 ms = 32 GB/s, the socket's STREAM bandwidth), while PCIe would allow 25 ms and the V100 computes
  the call in 12 ms. The model is conservative for mid-size calls (it assumes 640 GFlop/s on the CPU,
  which small calls do not reach), so a few profitable calls stay on the CPU, never the reverse.
* **What does not pay:** thin updates - rank 128-512 (k), or few rows (m = 250 stripes) - move too many
  bytes per flop: 4000 x 4000 x 256 runs at 0.62x on the GPU, 250 x 1000 x 4000 at 0.57x; the model
  rejects them (predicted 0.59-0.65x).

## 3. Which calls of a 1e6 factorization can go to the GPU

The BLAS hook's profiler (`HS_BLAS_PROFILE`, route `all`, GPU off) on the main factorization and one
inertia count of each 1e6 case, on hsolve's nested-dissection order (`bench_factspeed_factor --ordering
10 --jobs main,a`; the P9 default for 3-D problems; `results/p9_factspeed.md` profiled SCOTCH orders),
grouped by the size classes of the dgemm calls made **outside** OpenMP parallel regions (MUMPS' fronts
above its L0 layer, one call at a time - the only calls the engine takes):

| case | factorization: wall s (TFlop) | dgemm outside parallel regions: s (TFlop, GFlop/s) | of it k >= 1024, min(m, n) >= 512: s (TFlop) | of it k 128-511 (panel updates): s (TFlop) | dgemm inside the L0 subtrees: thread-s (TFlop) |
|---|---|---|---|---|---|
| syn-mem27-100, main | 23.4 (6.72) | 12.37 (5.93, 479) | 5.19 (3.26) | 4.76 (1.67) | 45.0 (1.01) |
| syn-mem27-100, count | 22.0 (6.73) | 12.35 (5.93, 480) | 5.19 (3.26) | 4.76 (1.67) | 44.1 (1.02) |
| yee-pec-box-1e6, main | 19.3 (5.43) | 10.13 (4.96, 490) | 4.20 (2.66) | 3.48 (1.24) | 30.8 (0.67) |
| yee-pec-box-1e6, count | 18.1 (5.44) | 10.15 (4.96, 489) | 4.17 (2.66) | 3.53 (1.26) | 29.9 (0.67) |

* The eligible class is MUMPS' contribution-block update (`DMUMPS_FAC_T_LDLT`, dfac_front_aux.F): for
  each row block of the CB (KEEP(58) = 1000 rows when the CB has more than KEEP(57) = 2000 rows) one
  `dgemm('N','N', Block, columns right of it, NPIV)` on the front (all three operands with leading
  dimension NFRONT) - m 512-1023, n 1024-8191, k 1024-8191, at 583-679 GFlop/s on the 16 cores.
  These are **49% of a factorization's flops but only 22% of its time** (syn-mem27-100: 3.26 of 6.72
  TFlop, 5.2 of 23.4 s).
* The rest stays: the rank-128 updates of a front's remaining fully summed columns (k = KEEP(420) =
  128: 3.5-4.8 s at ~350 GFlop/s, memory-bound - on the GPU they would cost 32 bytes of host traffic
  per 256 flops, section 2), the 250-row triangle stripes of the CB (m = KEEP(218) = 250), everything in
  the L0 subtrees (one call per thread inside a parallel region: 22-23 GFlop/s per thread, small
  fronts), and the non-BLAS third of the factorization (panels, assembly, zeroing, copies).
* So even an infinitely fast GPU could save at most the eligible 5.2 s of a 23.4 s factorization
  (1.29x); at the engine's average rate on these calls (1.8-1.9x, section 4) ~2.5 s (1.12x); measured:
  1.7-2.0 s (1.08-1.10x).

## 4. Factorizations at equal pivot order (`tools/bench/gpu_factor_ab.py`)

`bench_factspeed_factor --jobs main,a,b` makes the driver's three factorizations of a nearest run
exactly as the driver does (the main factorization at sigma, then the certificate's two inertia-only
counts at the window ends a, b on the main pivot order; results/p9_factspeed.md §1) with the hook
preloaded; HS_GPU off / on interleaved, 5 repetitions (order reversed every other one), one saved
pivot order per case (hsolve's nested dissection, deterministic): every variant factorizes the same
tree with the same flops. Wall time per factorization, medians [min-max]:

**4.1 MUMPS' default block sizes** (`factorizations`, label `mumps`):

| case | factorization | GPU off s | GPU on s | speedup | inertia (negative pivots), every run |
|---|---|---|---|---|---|
| syn-mem27-100 | main (sigma) | 23.43 [23.36-23.55] | 21.69 [21.60-21.96] | **1.080** | 20003 |
| syn-mem27-100 | count a | 22.05 [21.96-22.39] | 20.09 [19.96-20.15] | **1.097** | 19997 |
| syn-mem27-100 | count b | 22.01 [21.95-22.09] | 20.02 [19.82-20.13] | **1.099** | 20009 |
| yee-pec-box-1e6 | main (sigma) | 19.33 [19.31-19.36] | 18.33 [18.12-18.59] | **1.055** | 337613 |
| yee-pec-box-1e6 | count a | 18.07 [18.06-18.11] | 16.81 [16.61-16.98] | **1.075** | 337610 |
| yee-pec-box-1e6 | count b | 18.17 [18.12-18.19] | 16.87 [16.73-17.02] | **1.077** | 337620 |

The three together: syn-mem27-100 67.49 -> 61.80 s (**1.092x**, -5.7 s), yee-pec-box-1e6 55.57 ->
52.01 s (**1.068x**, -3.6 s); min-max ranges never overlap. The inertia is the same in all 10 runs of
each factorization (and equal to the complete references' counts: 20003 / 337613 at sigma). The GPU's
part per run (medians; all three factorizations, set-up once per process):

| case | offloaded calls | GPU GFlop (of the MUMPS estimate) | GPU call s | CPU-model s of those calls | H2D / D2H GB | set-up s | pipeline thread-s: pack / slot wait / sync / add |
|---|---|---|---|---|---|---|---|
| syn-mem27-100 | 483 | 9553 (47%) | 7.77 | 14.93 | 48.0 / 16.1 | 0.27 | 8.97 / 3.05 / 1.28 / 3.51 |
| yee-pec-box-1e6 | 423 | 7435 (45%) | 6.47 | 11.62 | 39.2 / 13.4 | 0.32 | 7.49 / 2.49 / 1.04 / 2.90 |

* **The GPU computes 45-47% of the factorizations' flops** at an effective 1.15-1.23 TFlop/s (the CPU
  does them at ~600 GFlop/s): 1.8-1.9x per offloaded call on average - these are mid-size calls (141-161 per
  factorization, 18-20 GFlop each on average), where the per-call fill and drain of the pipeline and the
  host traffic (section 2) leave the V100's 7 TFlop/s idle most of the time (busy ~20% of the call).
* The saving (1.9 / 1.2 s per factorization) is ~0.5 s per factorization smaller than the offloaded
  calls' model CPU time minus their GPU time predicts (2.4 / 1.7 s; the model's 640 GFlop/s is close
  to the profile's 581-590 GFlop/s for this class). Not attributed - candidates are the CPU work
  around the calls running with colder caches (C is written by the host additions, not by the CPU
  BLAS) and the pipeline threads' wake-ups on the BLAS team's cores; not measured further.
* yee-pec-box-1e6 gains less: its fronts are smaller (largest 12475 vs 15000), so more of its work is
  in the L0 subtrees and the panel updates.

**4.2 With the P9 block sizes** (`HS_MUMPS_BLOCKING=p9`: the CB row blocks KEEP(58) = 2000 rows -
twice as much work per byte moved for the GPU - and panels of 256; label `p9`;
the source-built MUMPS that is becoming the default build uses these block sizes by default):

| case | factorization | GPU off s | GPU on s | speedup | inertia, every run |
|---|---|---|---|---|---|
| syn-mem27-100 | main (sigma) | 22.21 [22.14-22.34] | 20.22 [20.18-20.46] | **1.098** | 20003 |
| syn-mem27-100 | count a | 20.75 [20.71-20.87] | 18.56 [18.47-18.73] | **1.118** | 19997 |
| syn-mem27-100 | count b | 20.74 [20.68-20.86] | 18.59 [18.43-18.77] | **1.116** | 20009 |
| yee-pec-box-1e6 | main (sigma) | 18.49 [18.46-18.54] | 17.34 [17.20-17.65] | **1.066** | 337613 |
| yee-pec-box-1e6 | count a | 17.24 [17.20-17.27] | 15.74 [15.69-15.90] | **1.095** | 337610 |
| yee-pec-box-1e6 | count b | 17.31 [17.29-17.39] | 15.92 [15.90-15.98] | **1.087** | 337620 |

The three together: syn-mem27-100 63.70 -> 57.37 s (**1.110x**, -6.3 s), yee-pec-box-1e6 53.04 ->
49.00 s (**1.082x**, -4.0 s); against MUMPS' default block sizes on the CPU (4.1) the two levers
together give 1.176x / 1.134x. GPU part per run:

| case | offloaded calls | GPU GFlop (of the estimate) | GPU call s | CPU-model s | H2D / D2H GB | set-up s | pack / slot wait / sync / add |
|---|---|---|---|---|---|---|---|
| syn-mem27-100 | 264 | 8763 (43%) | 5.70 | 13.69 | 27.4 / 15.2 | 0.28 | 4.27 / 4.04 / 0.44 / 3.21 |
| yee-pec-box-1e6 | 276 | 7011 (43%) | 5.19 | 10.96 | 24.2 / 13.8 | 0.28 | 3.81 / 3.69 / 0.47 / 2.79 |

With 2000-row blocks there are half as many calls, the B blocks are uploaded half as often (27 vs 48
GB per run) and the GPU reaches 1.35-1.54 TFlop/s effective, 2.1-2.4x per call. A screen with still
larger row blocks alone (KEEP(58) = 4000, KEEP(57) = 8000, panels unchanged; 1 run, syn-mem27-100,
screen2) made the CPU path slower (23.5 / 22.8 / 22.1 s for main / a / b) and the GPU path no faster
than with MUMPS' defaults (22.1 / 20.4 / 20.3 s against 21.7 / 20.1 / 20.0 in 4.1) - not pursued.

**4.3 Eight CPU threads** (one run each, MUMPS' default block sizes, syn-mem27-100,
`bench_factspeed_factor --threads 8` on the 16-core slot; label `t8`): main 28.44 -> 24.90 s
(1.142x), counts 27.08 -> 23.46 and 27.07 -> 23.38 s (1.154x / 1.158x); 82.6 -> 71.7 s for the three
(**1.151x**). The weaker the CPU, the more the same GPU work is worth (the model assumes 40 GFlop/s
per thread: here 333 calls, 8.2 TFlop offloaded). With 8 threads plus the GPU the factorizations are
within 6% of 16 threads alone (71.7 vs 67.5 s).

## 5. Whole runs (`tools/bench/gpu_e2e.py`)

Whole hsolve runs (CLI, the case's default configuration: nearest target, certificate exact, 16
threads, automatic memory budget from the 32 GiB cap), each through `compare.py`'s guarded, evaluated
run: **valid** = exit 0, success claimed, tools/eval passes at the case tolerance against the complete
reference. Solver time = report `time_total_s` (all of hs::solve). The GPU variant sets `HS_GPU`
(auto: the hook decides per factorization from MUMPS' estimate; on: every factorization).

### 5.1 n = 1e6 (5 interleaved repetitions)

off vs auto (at 1e6 auto offloads all three factorizations: their estimates are 5.4-6.7e12 >= 3e12),
interleaved (`whole_runs_1e6`; two slot calls: repetitions 1-3, 4-5). Medians;
phases from the report (exclusive: factor = analysis + main factorization, certificate = the two
exact counts, iterate = the Krylov iteration):

| case | variant | valid | solver s median [min-max] | speedup | factor s | certificate s | iterate s | peak GiB | certificate count | GPU factorizations | GPU GFlop (share) | eigenvalues vs off |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| syn-mem27-100 | off | 5/5 | 88.90 [88.20-89.15] | - | 27.30 | 44.76 | 16.19 | 13.10 | 12 | - | - | - |
| syn-mem27-100 | auto | 5/5 | 83.12 [82.35-83.34] | **1.069** | 25.46 | 40.73 | 16.20 | 13.58 | 12 | 3/3 | 9553 (0.47) | 3.3e-15 |
| yee-pec-box-1e6 | off | 5/5 | 73.76 [73.63-74.07] | - | 23.04 | 36.96 | 13.27 | 11.30 | 10 | - | - | - |
| yee-pec-box-1e6 | auto | 5/5 | 69.71 [69.55-70.11] | **1.058** | 21.97 | 34.14 | 13.20 | 11.75 | 10 | 3/3 | 7468 (0.46) | 2.4e-15 |

* **Every run valid** (tools/eval pass at the case tolerance, complete against the reference), the
  same certificate count in all 10 runs of a case, eigenvalues equal to 3e-15 relative; the ranges do
  not overlap. Whole runs gain **1.069x / 1.058x** (-5.8 / -4.1 s): exactly the factorizations'
  saving (factor -1.8 / -1.1 s, certificate -4.0 / -2.8 s); the iteration (block solves with the
  factors in host memory) is unchanged.
* Cost: the set-up (0.27-0.29 s, once per run, inside the first factorization) and +0.45-0.48 GiB of
  peak RSS (CUDA context, 224 MB of pinned staging); the GPU holds 1 GiB plus its context.
* Relative to the run: the offloaded calls take 6.5-7.9 s of a 70-83 s run (they would take 11.7-14.9
  s on the CPU by the model), and within them the V100 computes for ~1.2-1.5 s (7.5-9.6 TFlop).

### 5.2 Smaller problems

off / auto / on, 3 interleaved repetitions (`whole_runs_small`; the 1e5 3-D cases of the
head-to-head plus the heavier syn-mem125-46 and the 5.1e5-unknown syn-mem27-80):

| case | n | MUMPS estimate per factorization | variant | valid | solver s median [min-max] | vs off | GPU: calls, GFlop (share) | peak GiB |
|---|---|---|---|---|---|---|---|---|
| syn-mem27-46 | 9.7e4 | 6.6e10 | off | 3/3 | 2.04 [2.04-2.10] | - | - | 0.85 |
| | | | auto | 3/3 | 2.04 [2.02-2.04] | 0.997 | not used | 0.85 |
| | | | on | 3/3 | 2.31 [2.30-2.33] | 0.880 | 0, 0 | 1.27 |
| syn-mem125-46 | 9.7e4 | 4.7e11 | off | 3/3 | 10.77 [10.74-10.78] | - | - | 2.54 |
| | | | auto | 3/3 | 10.79 [10.75-10.82] | 0.998 | not used | 2.57 |
| | | | on | 3/3 | 11.05 [11.04-11.07] | 0.974 | 15, 140 (10%) | 3.00 |
| yee-pec-box-1e5 | 1.0e5 | 6.0e10 | off | 3/3 | 2.49 [2.46-2.50] | - | - | 0.77 |
| | | | auto | 3/3 | 2.47 [2.47-2.47] | 1.007 | not used | 0.76 |
| | | | on | 3/3 | 2.75 [2.75-2.77] | 0.904 | 0, 0 | 1.20 |
| ss-qa8fk-qa8fm | 6.6e4 | 1.8e10 | off | 3/3 | 1.52 [1.50-1.52] | - | - | 0.69 |
| | | | auto | 3/3 | 1.51 [1.50-1.51] | 1.006 | not used | 0.72 |
| | | | on | 3/3 | 1.76 [1.76-1.79] | 0.861 | 0, 0 | 1.17 |
| syn-mem27-80 | 5.1e5 | 1.8e12 | off | 3/3 | 32.10 [32.05-32.11] | - | - | 6.19 |
| | | | auto | 3/3 | 32.02 [32.01-32.08] | 1.003 | not used | 6.21 |
| | | | on | 3/3 | 31.84 [31.59-31.85] | 1.008 | 147, 1679 (32%) | 6.65 |

The 1-run screen of the same build (`size_screen`, off vs on) adds syn-mem27-64 (n = 2.6e5,
4.6e11 flops per factorization: 0.97x, 175 GFlop offloaded), ss-2cubes_sphere 0.85x, ss-tem152078
0.93x and yee-pec-cube-32 0.72x (nothing offloaded).

* Up to n ~ 1e5 the factorizations' fronts are too small: the model offloads nothing or almost nothing,
  and `on` only pays the set-up (0.25-0.36 s: CUDA context, two cuBLAS handles, 1 GiB device and 224 MB
  pinned buffers, threads) - 0.86-0.97x. At n = 5.1e5 the offload computes a third of the flops and
  breaks even (1.008x); the gain at 1e6 is section 5.1.
* **`auto` is the CPU path below its threshold**: identical times (0.997-1.007x), the same counts, no
  CUDA library loaded (`"available": null` in the report) - the threshold of 3e12 estimated flops per
  factorization sits between syn-mem27-80 (1.8e12) and the 1e6 cases (5.4-6.7e12).
* Every run of every variant valid, the same certificate counts in all 9 runs of a case.

## 6. Division of labour (for the "并行方案" section)

What runs where in a nearest run with `--gpu auto` (or on) on a machine with one GPU, 16 cores:

| phase | where | how | share at n = 1e6 (syn-mem27-100) |
|---|---|---|---|
| loading, norm estimates, B checks | CPU, 16 threads | as without GPU | small |
| analysis: ordering (own nested dissection), MUMPS symbolic | CPU, 16 threads (OpenMP tasks) | as without GPU | ~3.9 s (ordering 2.5 s) |
| numeric LDLᵀ: L0 subtrees (small fronts, one per thread) | CPU, 16 threads | OpenBLAS, single-threaded calls | ~6.5-7.5 s per factorization |
| numeric LDLᵀ above L0: panels, assembly, rank-128 updates, triangle stripes | CPU, 16 threads | OpenBLAS threaded per call, MUMPS OpenMP loops | ~11 s per factorization |
| numeric LDLᵀ above L0: contribution-block updates (k >= ~1000) | **GPU** (FP64 cuBLAS) + 3 host threads (2 staging, 1 adding) | hook route `Gpu`; CPU threads wait | 5.2 s of CPU time per factorization -> 2.6 s on the GPU path (3.2 TFlop, 161 calls) |
| the certificate's two inertia-only counts | as the numeric LDLᵀ (same split) | same | 2 of the 3 factorizations |
| Krylov iteration: MUMPS block solves, orthogonalization | CPU, 16 threads | BLAS hook routes (P5) | 16.2 s |
| residual checks, Rayleigh-Ritz | CPU | | small |

Measured speedups of the GPU part (syn-mem27-100, n = 1e6, nearest k = 12, certificate exact; the
three factorizations at equal pivot order, sections 4.1-4.3; whole runs, section 5.1):

| configuration | three factorizations s | speedup | whole run s | speedup |
|---|---|---|---|---|
| 16 threads (MUMPS default block sizes) | 67.5 | 1 | 88.9 | 1 |
| 16 threads + GPU (`--gpu auto`) | 61.8 | 1.092 | 83.1 | 1.069 |
| 16 threads, P9 block sizes | 63.7 | 1.060 | - | - |
| 16 threads, P9 block sizes + GPU | 57.4 | 1.176 | - | - |
| 8 threads | 82.6 | 0.817 | - | - |
| 8 threads + GPU | 71.7 | 0.941 (1.151 over 8 threads) | - | - |

(The rows come from the campaigns of 4.1 / 4.2 / 4.3 - same node, build and pivot order, different
slot calls; the 8-thread rows are single runs.) yee-pec-box-1e6: factorizations 1.068x (P9 block
sizes 1.082x), whole run 1.058x. At n <= 2.6e5 the
GPU has nothing worth offloading (section 5.2) - `auto` keeps those runs on the CPU.

Proposed text for docs/SOLUTION.md §6 (replacing its last line, "GPU：赛题允许但非必需……"):

> GPU（P10，`results/p10_gpu.md`）：BLAS hook 的 GPU 路由（`src/gpu`）在 MUMPS 分解（含证书的两次精确计数）中把
> L0 层之上、主机流量模型判为划算的大 dgemm（前沿贡献块的更新，k ≥ ~1000）以 FP64 cuBLAS 卸载到一张 GPU，CUDA 库在
> 运行时 dlopen（构建不依赖 CUDA）。分工：分析与排序、L0 子树、面板与组装、回代与 Krylov 迭代都在 CPU（16 线程）；
> GPU 承担 1e6 分解约 43–47% 的 flops（只占 CPU 时间的 22%），每次调用 1.7–5.0 倍；1e6 三维测例上三次分解 1.07–1.09 倍（配 P9 块参数 1.08–1.11 倍），整次运行 1.058–1.069 倍（证书 exact，每次有效、计数相同）。受主机内存带宽
> （暂存与加回的流量，约 30 GB/s）限制，1e5 规模没有值得卸载的调用（`on` 为 0.72–0.97 倍），所以 `--gpu auto` 只在分解
> 估计 ≥ 3e12 flops 时用 GPU；默认 `off`。无 GPU / 无 CUDA 库 / 显存不足时自动为 CPU 路径并在报告中说明（降级模式，
> 评测机即此情形）；惯性计数与 CPU 路径完全相同。

## 7. Degrade mode and tests

* **Default: off.** A build's GPU mode is `off` unless `HS_GPU` / `hsolve --gpu` / the CMake cache
  `HSOLVE_GPU_DEFAULT` say otherwise; then no CUDA library is even opened and every call takes
  exactly today's CPU path (the router's only addition is one atomic load per dgemm). The judging
  machine needs nothing: the binary has no CUDA dependency (`ldd` shows none; the libraries are
  dlopen'ed), so the same build runs there.
* **auto / on without a GPU** (no driver, no `libcudart.so.12`/`libcublas.so.12`, no visible device,
  too little free device memory): the first factorization that wants the GPU fails to create the
  engine, the reason goes into the report (`"available": false, "reason": ...`), and the process
  runs the CPU path - the same factorizations, counts and eigenpairs. `auto` below its threshold does
  not look for CUDA at all (`"available": null`). A CUDA error during an offloaded call (tested with
  an injected fault) completes the call on the CPU - the tiles not yet added are recomputed - and
  returns the process to the CPU for good.
* **Tests** (ctest, `test_gpu` / `test_gpu_hook`, 14 tests): a plain ctest run sees no GPU (the test
  processes set `CUDA_VISIBLE_DEVICES` to the empty list before main - nothing touches the shared
  node's GPUs) and checks the degrade paths: no engine without a device, auto below the threshold never
  loads CUDA, mode on degrades with a reason, the hooked MUMPS factorization of an indefinite 3-D
  Laplacian gives the exact Sylvester count with the GPU "on", and `hsolve --gpu on` gives the CPU
  run's eigenpairs and certificate count with `"available": false` in the report. With
  `HSOLVE_TEST_GPU_DEVICE=7` (through `gpu_slot.sh`) the same tests run on the GPU, plus the engine
  against the CPU BLAS (every transpose pair, both split directions, alpha / beta incl. beta = 0 with
  NaN in C, padded leading dimensions, 1 MiB slots for many tiles, GPU-only and hybrid), bitwise
  reproducibility of GPU-only calls, the offload model's choices, concurrent callers (one is served,
  the others get "busy" and compute on the CPU) and the injected fault. Both configurations pass
  (section 9): 14/14 tests with GPU 7 (12 run, the 2 no-GPU tests skipped) and without (7 run, the 7
  GPU tests skipped); a build with `-DHSOLVE_GPU=OFF` compiles warning-free and skips the hook tests.
  The whole suite passes in the lane without a GPU on a fresh build (0 warnings; ctest 1023/1023,
  pytest 378).
* **Python**: the extension module has no hook; with `libhsolve_blashook.so` preloaded and
  `HS_GPU=on` (forced to offload everything, `HS_GPU_OPTS=min_speedup=0,min_gflop=0`) a
  `hsolve.solve` of syn-mem27-46 offloaded 95% of its factorization flops on GPU 7 and succeeded with
  the CPU path's certificate count (the report at exit through `HS_GPU_REPORT`); not timed.

## 8. Not done, and why

Measured negative or bounded (numbers in sections 2-5):

* **Hybrid CPU + GPU within a call** (the CPU computes chunks from the other end with 4-14 BLAS
  threads, both sides claiming guided portions): slower than GPU-only on every large shape (2000 x 8000
  x 4000: 3.95x / 4.13x with 13 / 8 CPU threads vs 5.04x GPU-only; 1000 x 9000 x 4000: 2.83x vs 3.08x) -
  the CPU's BLAS (its packing) and the staging compete for the same ~32 GB/s of host memory bandwidth;
  it only wins on shapes the model keeps on the CPU anyway. Kept as a measurement option (`hybrid=1`).
* **Letting the copy engine read staging data from the LLC** (pack ~1 MB chunks with regular stores
  and copy each at once, so that DDIO could serve the DMA reads from cache): 0.89-1.03x of whole-tile
  packing with non-temporal stores. Option `chunk_kb` / `nt`, off.
* **Registering MUMPS' memory** instead of staging (`cudaHostRegister`, 14-15 GB/s of address range):
  the operands are blocks of a front with leading dimension NFRONT, so a B tile of k rows spans
  NFRONT / k (4-30x) more address range than data; registering per call costs more than staging, and
  keeping registrations across calls is unsafe (MUMPS frees and reuses front memory; a stale
  registration DMAs from the old pages).
* **Caching operands on the GPU across calls**: successive row-block calls of one front share most of
  B (the U block), which would cut the uploads several-fold, but the hook cannot prove the host data
  unchanged between calls (same address, different front later) - not sound, not done.
* **Equal tiles instead of guided ones** for GPU-only calls: 5-10% slower (longer pipeline drain;
  development screen, 3 runs, 5 MUMPS-like shapes).
* **Calls inside MUMPS' L0 subtrees** (one call per thread at 22-35 GFlop/s per thread): the engine
  serves one call at a time and its pipeline threads would oversubscribe the 16 busy cores; the
  eligible L0 calls are a few thread-seconds per factorization - estimated <= 2% of a factorization.
* **Larger contribution-block row blocks for the GPU** (KEEP(58) = 4000 alone): fewer, larger calls,
  but the CPU part gets slower and the GPU path ends no faster than with MUMPS' defaults (section 4.2);
  the P9 block sizes (2000 rows, panels of 256) are the better setting with and without the GPU.
* **The solves and the Krylov iteration**: the solves stream the factors from host memory
  (memory-bound, 16-30 GB/s, docs/BLAS.md section 2); over PCIe 3 (12.8 GB/s) they can only be slower
  unless the factors live on the GPU, which needs a GPU solver of its own - SPRAL SSIDS was measured
  in G1 (branch `exp-gpu-ssids`, results/gpu_g1.md): factorization 0.9-1.85x at 1e5, block solves
  1.6-2.9x slower, the 16 GB card full at 1e6. The iteration's own dense work (Rayleigh-Ritz,
  orthogonalization) is 1-2 s of a 1e6 run. Not pursued.
* **Mixed precision on the GPU** (FP32 / tensor cores): the factorization's inertia must be that of
  K in double precision (the exact certificate); P6 found single-precision factors unhelpful for the
  shift-invert iteration as well (results of branch `p6-mixprec-fix`). Every GPU operation here is
  FP64.

## 9. Reproduction

Everything through `tools/dev/cluster_run.sh gpu02 [--allow-dirty] -- ...` (env/run.sh activates the
environment); timed runs from the measurement work directory (`HS_CLUSTER_NAME=p10gpu-meas`, built once
at ce76a37), `D=$HOME/p7_p7-membudget/data`, `S="tools/bench/gpu_slot.sh 7"` (checks that GPU 7 is idle,
`CUDA_VISIBLE_DEVICES=7`, `guard.py --exclusive --timeout T`):

```bash
tools/dev/build.sh release                                   # lane; nothing CUDA needed
# section 2: machine and engine
M=results/raw/p10gpu/micro.jsonl                             # one slot call (sh -c) for all four
$S 900 -- build/release/bench_gpu_dgemm info > $M
$S 900 -- build/release/bench_gpu_dgemm copy --mb 16,64,256 --reps 5 >> $M
$S 900 -- build/release/bench_gpu_dgemm devgemm --reps 5 --shapes \
   1000:9000:4000,1000:4000:2048,2000:8000:4000,1000:9000:512,250:1000:4000,4000:4000:128,4000:4000:256,8192:8192:8192 >> $M
$S 900 -- build/release/bench_gpu_dgemm gemm --ld 15800 --slot-mb 16 --reps 3 \
   --modes cpu,gpu,gpu-p1,gpu-p3,gpu-s3,gpu-m32,gpu-nt0,gpu-k1024-nt0,hyb,hyb-c8 --shapes \
   1000:9000:4000,2000:8000:4000,1000:4000:2048,1000:2000:2048,1000:9000:1024,1000:9000:512,4000:4000:256,250:1000:4000 >> $M
python3 tools/bench/gpu_dgemm_table.py $M
# section 3: profile (GPU off), per case C with its window --lower/--upper (gpu_factor_ab.py WINDOWS)
python3 tools/run/guard.py --exclusive --timeout 1200 -- env LD_PRELOAD=$PWD/build/release/libhsolve_blashook.so \
   HS_BLAS_ROUTE=all HS_BLAS_PROFILE=prof_C.json build/release/bench_factspeed_factor --case $D/cases/C \
   --lower LO --upper HI --ordering 10 --jobs main,a
build/release/bench_asmkern_report --top 30 prof_C.json
# section 4: factorizations, 5 interleaved repetitions, GPU off / on, saved ND pivot order per case
$S 2700 -- python3 tools/bench/gpu_factor_ab.py --out results/raw/p10gpu/fact-1e6.jsonl --reps 5 --label mumps \
   syn-mem27-100 yee-pec-box-1e6
$S 2700 -- python3 tools/bench/gpu_factor_ab.py --out results/raw/p10gpu/fact-1e6-p9.jsonl --reps 5 \
   --env HS_MUMPS_BLOCKING=p9 --label p9 syn-mem27-100 yee-pec-box-1e6
$S 2700 -- python3 tools/bench/gpu_factor_ab.py --out results/raw/p10gpu/fact-1e6-t8.jsonl --reps 1 \
   --bench-args "--threads 8" --label t8 syn-mem27-100
# section 5: whole runs, evaluated (valid = exit 0, success, tools/eval pass); reps split over slot calls
$S 2700 -- python3 tools/bench/gpu_e2e.py --tag p10-e2e-1e6 --reps 3 --variants off,auto --root $D --mem-gb 32 \
   --timeout 1500 syn-mem27-100 yee-pec-box-1e6        # then the same with --reps 2 (repetitions 4-5)
$S 2700 -- python3 tools/bench/gpu_e2e.py --tag p10-e2e-small --reps 3 --variants off,auto,on --root $D \
   --mem-gb 32 syn-mem27-46 syn-mem125-46 yee-pec-box-1e5 ss-qa8fk-qa8fm syn-mem27-80
$S 2400 -- python3 tools/bench/gpu_e2e.py --tag p10-sizes --reps 1 --variants off,on --root $D syn-mem27-46 \
   syn-mem125-46 yee-pec-box-1e5 ss-qa8fk-qa8fm syn-mem27-64 syn-mem27-80          # screen, section 5.2
# section 7: tests
tools/dev/test.sh release                                   # lane, no GPU visible to the tests
$S 900 -- env HSOLVE_TEST_GPU_DEVICE=7 ctest --preset release --no-tests=error -R test_gpu --output-on-failure
```
