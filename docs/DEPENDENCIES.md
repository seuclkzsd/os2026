# 依赖清单与许可证

赛题要求「说明开源依赖、授权和调用方式」，且不得以不可验证的闭源服务为核心能力。
新增依赖必须在此登记（名称、版本、许可证、用途、调用方式）。

| 依赖 | 版本 | 许可证 | 用途 | 调用方式 |
|---|---|---|---|---|
| MUMPS | 5.8.2（conda-forge mumps-seq，OpenMP） | CeCILL-C | 稀疏 LDLᵀ / LU 分解、惯性、多右端回代；单/双精度 | C 接口 `dmumps_c/zmumps_c/smumps_c/cmumps_c`，`src/factor/` |
| METIS | 5.1.0 | Apache-2.0 | MUMPS 的填充约简排序 | 经 MUMPS（ICNTL(7)） |
| SCOTCH | 7.0 | CeCILL-C | MUMPS 备选排序 | 经 MUMPS |
| OpenBLAS | 0.3.34（OpenMP 版） | BSD-3-Clause | BLAS / LAPACK | Fortran 符号，`src/linalg/blas_lapack.hpp` |
| zlib | 1.3 | zlib | 解压 `.npz` | `src/io/` |
| nlohmann/json | 3.12 | MIT | 报告与日志 JSON | 头文件库 |
| CLI11 | 2.7 | BSD-3-Clause | 命令行解析 | 头文件库，`apps/` |
| GoogleTest | 1.18 | BSD-3-Clause | C++ 单元测试 | 仅测试 |
| pybind11 | 3.1 | BSD-3-Clause | Python 接口（P2 起） | `python/` |
| NumPy / SciPy | 2.4 / 1.17 | BSD-3-Clause | 测例生成、评测、基线对比 | 仅 `tools/`，不进入求解核心 |
| GCC / gfortran | 13.4 | GPL-3.0 + 运行库例外 | 编译 | 构建工具 |

不使用：Intel MKL / PARDISO、HSL（MA57/MA97）、Panua-Pardiso 等非开源组件。
CUDA（若在 P3 引入）为 NVIDIA 闭源但免费，赛题明确允许；仅作可选加速，不作为核心能力。

公开测试矩阵（SuiteSparse Matrix Collection）按 CC-BY 4.0 使用，引用见 `tools/gen/` 生成的元数据。
