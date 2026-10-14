# 依赖清单与许可证

赛题要求「说明开源依赖、授权和调用方式」，且不得以不可验证的闭源服务为核心能力。
新增依赖必须在此登记（名称、版本、许可证、用途、调用方式）。各组件的版权声明与许可证要求复制的文本
集中在根目录的 `THIRD_PARTY_NOTICES`（分发 hsolve 的二进制时随附）。

## 求解器（环境 `os2026`，`env/environment.yml`）

| 依赖 | 版本 | 许可证 | 用途 | 调用方式 |
|---|---|---|---|---|
| MUMPS | 5.8.2（conda-forge mumps-seq，OpenMP） | CeCILL-C | 稀疏 LDLᵀ / LU 分解、惯性、多右端回代；单/双精度 | C 接口 `dmumps_c/zmumps_c/smumps_c/cmumps_c`，`src/factor/` |
| METIS | 5.1.0 | Apache-2.0 | MUMPS 的填充约简排序 | 经 MUMPS（ICNTL(7)）；不直接链接 |
| SCOTCH | 7.0 | CeCILL-C | MUMPS 备选排序 | 经 MUMPS；不直接链接 |
| OpenBLAS | 0.3.34（OpenMP 版） | BSD-3-Clause | BLAS / LAPACK；BLAS hook 的窄核改编自其 SkylakeX 核（见下） | Fortran 符号，`src/linalg/blas_lapack.hpp`；MUMPS 的 BLAS 调用经 `src/blas` 的 hook（默认链入 hsolve） |
| zlib | 1.3 | zlib | 解压 `.npz` | `src/io/` |
| nlohmann/json | 3.12 | MIT | 报告与日志 JSON | 头文件库 |
| CLI11 | 2.7 | BSD-3-Clause | 命令行解析 | 头文件库，`apps/` |
| GoogleTest | 1.18 | BSD-3-Clause | C++ 单元测试 | 仅测试 |
| pybind11 | 3.1 | BSD-3-Clause | Python 接口（P2 起） | `python/` |
| NumPy / SciPy | 2.4 / 1.17 | BSD-3-Clause | 测例生成、评测、基线对比 | 仅 `tools/`，不进入求解核心 |
| GCC / gfortran | 13.4 | GPL-3.0 + 运行库例外 | 编译 | 构建工具 |

P4 未增加求解器依赖：
- 自研并行嵌套剖分（`src/factor/ordering*.cpp`）自成一体，以 PERM_IN 交给 MUMPS；METIS / SCOTCH 只是
  libdmumps 的间接依赖，hsolve 不直接链接也不调用它们（`docs/DESIGN.md` §3.4）。
- AVX-512 稀疏核用编译器内建函数（`target` 属性）与 `__builtin_cpu_supports` 运行时检测，无额外库；
  无 AVX-512 的 CPU 上自动走可移植内核（`docs/DESIGN.md` §3.2）。

P5 未增加求解器依赖：
- **BLAS hook**（`src/blas`，`docs/BLAS.md`、`docs/DESIGN.md` §3.8）：拦截 MUMPS 与 hsolve 的
  `dgemm_` / `dtrsm_` / `dgemv_`，按形状改道或原样转发给环境的 OpenBLAS；只用 OpenMP 与 libdl
  （`dlsym(RTLD_NEXT)` 找真实例程），AVX-512 / AVX2 核用编译器内建函数并由 CPUID 在运行时选择。窄核的结构
  **改编自 OpenBLAS 0.3.34**（`kernel/x86_64/dgemv_n_microk_skylakex-4.c`，Copyright (c) 2014 The OpenBLAS
  Project；`kernel/x86_64/dgemm_small_kernel_{nn,tn}_skylakex.c`，Copyright (c) 2021 The OpenBLAS Project），
  BSD-3-Clause：源码分发所需的声明在 `src/blas/simd_kernels.hpp`（`narrow_kernels.cpp` 指向它），二进制分发
  所需的声明在 `THIRD_PARTY_NOTICES` 与 `docs/BLAS.md` §7。小三角求解核按参考 BLAS 的循环顺序自写。
  链入方式：CMake 选项 `HSOLVE_BLAS_HOOK`（默认 ON，链入 `hsolve` 可执行文件）或 `libhsolve_blashook.so`
  （LD_PRELOAD）；libhsolve 与 Python 扩展模块不带。
- 自适应排序把 hsolve 自研的嵌套剖分列为候选、config 可用 `ordering = 10`（`docs/DESIGN.md` §3.6-3b）：仍是
  自己的代码，METIS / SCOTCH / PORD 仍只经 MUMPS 调用。
- 只用于实验、不随 hsolve 使用或分发：`tools/deps/build_openblas.sh`、`build_blis.sh`、`build_libxsmm.sh`
  从固定版本的源码包（sha256 校验；脚本头注明下载地址）在集群节点上编译 OpenBLAS 0.3.34、BLIS 2.1、
  LIBXSMM 1.17（均为 BSD-3-Clause），作为 LD_PRELOAD 替换件评估 BLAS（结论：都不采用，`docs/BLAS.md` §3、§6）。

P10 多进程模式（可选，`docs/PARALLEL.md`、`docs/DESIGN.md` §3.9）：
- **`hsolve-mpi`**（CMake `HSOLVE_WITH_MPI=AUTO`：找到 MPI 就构建）动态链接 MPI：集群节点上为 os2026-ref 环境的
  **Open MPI 5.0.11**（BSD-3-Clause），经它链接 PMIx 5.0.8（BSD-3-Clause）、hwloc 2.13（BSD-3-Clause；经它
  libxml2 2.15，MIT，与 GNU libiconv 1.18，LGPL-2.1，动态链接）、libevent 2.1.12（BSD-3-Clause）、UCX 1.22 与
  UCC 1.8（BSD-3-Clause）、libfabric 1.22（BSD-2-Clause / GPL-2.0 双许可，按 BSD-2-Clause 使用）、rdma-core 65
  （同为 BSD-2 / GPL-2 双许可）、libnl 3.11（LGPL-2.1+，动态链接）；启动脚本 `hsolve-mpirun` 调用该 MPI 的 `mpiexec`
  （Open MPI 自带的 PRRTE `prterun`）。每个进程仍链接顺序版 MUMPS（其 libmpiseq 的 MPI 桩函数由排在前面的真
  MPI 覆盖，DESIGN §3.9）。版权声明与许可证：`THIRD_PARTY_NOTICES` 3.7。`hsolve` 可执行文件、libhsolve 与 Python
  模块不链接 MPI；评测机没有 MPI 时不构建 `hsolve-mpi`，普通 `hsolve` 即默认路径。
- 只测过 Open MPI 5.0.11；启动脚本经环境变量把布局传给各 rank，Open MPI 专有的选项（`-x`、`--bind-to none`、测试的
  `--oversubscribe`）只在 `mpiexec --version` 表明是 Open MPI 时才加（MPICH / Intel MPI 的 Hydra 默认转发环境、
  不绑核、不限 rank 数；其它启动器的选项可经 CMake 的 `MPIEXEC_PREFLAGS` 加给测试）。

不使用：Intel MKL / PARDISO、HSL（MA57/MA97）、Panua-Pardiso 等非开源组件。
CUDA（若引入）为 NVIDIA 闭源但免费，赛题明确允许；仅作可选加速，不作为核心能力。

公开测试矩阵（SuiteSparse Matrix Collection）按 CC-BY 4.0 使用，引用见 `tools/gen/` 生成的元数据。

## 对照基线（环境 `os2026-ref`，用于 `tools/bench/`；其 Open MPI 也是集群节点上 `hsolve-mpi` 链接的 MPI，见上）

| 依赖 | 版本 | 许可证 | 用途 |
|---|---|---|---|
| PETSc / SLEPc（petsc4py / slepc4py） | 3.26 | BSD-2-Clause | SLEPc Krylov-Schur + 位移反演基线（`tools/bench/baseline_slepc.py`） |
| MUMPS（MPI 版） | 5.8.2 | CeCILL-C | 基线的分解（与 hsolve 同版本，使对比只差特征值层） |
| OpenMPI | 5.0.11 | BSD-3-Clause | 基线的多 rank 布局；`hsolve-mpi` 的 MPI（P10） |
| libfabric | 1.22（集群节点，见下） | BSD-2-Clause / GPL-2.0 双许可 | OpenMPI 的传输层 |
| PT-SCOTCH / ParMETIS | — | CeCILL-C / 明尼苏达大学许可（限非营利研究） | 基线调优中试过的并行分析（ICNTL(28)=2） |

- 该环境的 PETSc 是**实数**构建：SLEPc 不能运行 10 个复厄密测例，对比只覆盖实数测例（需要复数标量的
  PETSc/SLEPc 环境才能补全）。
- `tools/bench/slepc_profiles.json` 存放调优配置（`results/baseline_tuning.md`，目前只有 2 核）。

## 集群节点部署（gpu01 / gpu02 / c001–c005）

节点为 CentOS 7（glibc 2.17）、无外网。`tools/dev/deploy_cluster.sh HOST` 由开发服务器的 conda 包缓存
离线重建两个环境：
1. `conda list --explicit --md5` 导出 `os2026` 与 `os2026-ref` 的包列表，改写成指向节点上包目录的
   `file://` 列表；
2. 把包 rsync 到节点，`conda create -p $HOME/envs/<env> --offline --file <env>.local.explicit.txt`；
3. **SLEPc 环境换用 libfabric1 1.22**：conda-forge 当前的 libfabric1 2.x 需要 glibc 2.25，节点只有 2.17；
   1.22 只需 glibc 2.14，且提供 OpenMPI 5 用到的全部 FABRIC_1.x 符号版本（脚本把它下载进本机包缓存）；
4. 建裸仓库（git remote `cluster`）、克隆，并拷贝不入库的 `data/`。

节点侧设置（conda 位置、环境前缀、`HS_REF_ENV`、核与内存上限）在 `env/hosts/<host>.sh`，由
`env/activate.sh` 按短主机名加载；在节点上运行某个 worktree 的提交用 `tools/dev/cluster_run.sh`
（`docs/DESIGN.md` §6）。
