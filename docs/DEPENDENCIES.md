# 依赖清单与许可证

赛题要求「说明开源依赖、授权和调用方式」，且不得以不可验证的闭源服务为核心能力。
新增依赖必须在此登记（名称、版本、许可证、用途、调用方式）。

## 求解器（环境 `os2026`，`env/environment.yml`）

| 依赖 | 版本 | 许可证 | 用途 | 调用方式 |
|---|---|---|---|---|
| MUMPS | 5.8.2（conda-forge mumps-seq，OpenMP） | CeCILL-C | 稀疏 LDLᵀ / LU 分解、惯性、多右端回代；单/双精度 | C 接口 `dmumps_c/zmumps_c/smumps_c/cmumps_c`，`src/factor/` |
| METIS | 5.1.0 | Apache-2.0 | MUMPS 的填充约简排序 | 经 MUMPS（ICNTL(7)）；不直接链接 |
| SCOTCH | 7.0 | CeCILL-C | MUMPS 备选排序 | 经 MUMPS；不直接链接 |
| OpenBLAS | 0.3.34（OpenMP 版） | BSD-3-Clause | BLAS / LAPACK | Fortran 符号，`src/linalg/blas_lapack.hpp` |
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

不使用：Intel MKL / PARDISO、HSL（MA57/MA97）、Panua-Pardiso 等非开源组件。
CUDA（若引入）为 NVIDIA 闭源但免费，赛题明确允许；仅作可选加速，不作为核心能力。

公开测试矩阵（SuiteSparse Matrix Collection）按 CC-BY 4.0 使用，引用见 `tools/gen/` 生成的元数据。

## 对照基线（环境 `os2026-ref`，只用于 `tools/bench/`，不随 hsolve 分发）

| 依赖 | 版本 | 许可证 | 用途 |
|---|---|---|---|
| PETSc / SLEPc（petsc4py / slepc4py） | 3.26 | BSD-2-Clause | SLEPc Krylov-Schur + 位移反演基线（`tools/bench/baseline_slepc.py`） |
| MUMPS（MPI 版） | 5.8.2 | CeCILL-C | 基线的分解（与 hsolve 同版本，使对比只差特征值层） |
| OpenMPI | 5 | BSD-3-Clause | 基线的多 rank 布局 |
| libfabric | 1.22（集群节点，见下） | BSD-2-Clause / GPL-2.0 双许可 | OpenMPI 的传输层 |
| PT-SCOTCH / ParMETIS | — | CeCILL-C / 明尼苏达大学许可（限非营利研究） | 基线调优中试过的并行分析（ICNTL(28)=2） |

- 该环境的 PETSc 是**实数**构建：SLEPc 不能运行 10 个复厄密测例，对比只覆盖实数测例（需要复数标量的
  PETSc/SLEPc 环境才能补全）。
- `tools/bench/slepc_profiles.json` 存放调优配置（`results/baseline_tuning.md`，目前只有 2 核）。

## 集群节点部署（gpu01 / gpu02 / c001）

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
