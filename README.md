# hsolve — 大规模非正定厄密稀疏矩阵特征值求解器

面向电磁仿真（微波器件、天线、电磁兼容、光子晶体、波导模式分析）的特征值求解器：
支持 `A x = λ x` 与 `A x = λ B x`（B 正定或**不定**），矩阵为实对称或复厄密、规模 ~1e5、
非正定；求目标位移 σ 附近 / 区间内 / 谱两端的特征对。核心为 C++17 + MUMPS（对称不定
LDLᵀ、惯性计数）+ 自研 Krylov-Schur（含块模式）、LOBPCG、Jacobi-Davidson；B 正定（或无 B）时结果带完备性证书
（Sylvester 惯性计数，或在已有分解上的随机排除检验 / 单精度计数）；命令行与 Python 两种接口。

- 赛题：`商飞赛题_非正定厄密稀疏矩阵特征值求解.md`
- 设计与模块契约：`docs/DESIGN.md`
- 依赖与许可证：`docs/DEPENDENCIES.md`、`THIRD_PARTY_NOTICES`
- 方案说明（交付物 a）：`docs/SOLUTION.md`；文件格式：`docs/FORMATS.md`
- Python 接口（scipy/numpy 进出）：`docs/PYTHON.md`
- MUMPS 之下的稠密 BLAS（剖析、替换库、BLAS hook）：`docs/BLAS.md`
- 并行方案（多线程、多进程 `hsolve-mpi`、GPU 卸载 `--gpu`；分工与实测加速比）：`docs/SOLUTION.md` §6、`docs/PARALLEL.md`
- 实测汇总：`results/p9_measurements.md`（P9 集成的最终测量：1e6 与 1e5 对 main a831d79、对调优 SLEPc、
  1–16 线程的加速比，摘要见 `docs/SOLUTION.md` §6、§9.1）；P9 的三个实验 `results/p9_ordering.md`（自研嵌套剖分）、
  `results/p9_factspeed.md`（1e6 的数值分解、MUMPS 块参数与自编 MUMPS）、`results/p9_mpi.md`（1e6 对 SLEPc、
  MPI MUMPS 的上界：负结果）；P10 的 `results/p10_mpi.md`（每 socket 一个 helper 进程的多进程模式）、`results/p10_gpu.md`
  （cuBLAS 卸载 1e6 分解的大 dgemm）；负结果记录 `results/gpu_g1.md`（SPRAL SSIDS on GPU）、`results/p6_mixprec.md`（单精度分解）；更早的 `results/p8_certcost.md`、`results/p7_membudget.md`、
  `results/p5c_measurements.md`、`results/p3_measurements.md`、`results/p4_measurements.md`；SLEPc 调优与公平对比：
  `results/baseline_tuning.md`、`results/baseline_tuning_p16.md`、`results/cmp-p9-1e6-profiles.json`（1e6）；
  测量程序：`bench/README.md`

## 快速开始

```bash
conda env create -f env/environment.yml      # 一次性
source env/activate.sh                       # 激活环境并设置资源策略默认值
tools/dev/build.sh release                   # 构建 (build/release)：先编译 MUMPS（首次约 1.5 分钟，之后复用），再编译 hsolve
tools/dev/test.sh release                    # C++ 单元测试 + Python 测试（工具与接口）
```

开发服务器自 P4 收尾起只做编辑与 git，编译、测试与实验都在集群节点上运行（见下「集群节点」）。

**MUMPS 的来源**（P9 集成）：`tools/dev/build.sh` 默认（`HS_MUMPS=source`）由仓库内的官方源码包
`third_party/mumps/MUMPS_5.8.2.tar.gz`（sha256 校验，CeCILL-C，离线可构建）加 `patches/mumps-5.8.2-l0-blocking.patch`
编译顺序版 MUMPS 5.8.2（conda-forge 的可移植编译选项，链接环境的 METIS / SCOTCH / OpenBLAS；`tools/dev/mumps_cache.sh`
按源码包、补丁、选项、脚本、编译器与环境的哈希缓存在 `build/deps/`，`HS_MUMPS_CACHE` 可改），所有预设、命令行与
Python 模块都链接它（`-DHSOLVE_MUMPS_ROOT`，RPATH 在环境之前），MUMPS 的块参数默认取 P9 的（分解快 5–8%，
`results/p9_factspeed.md`）。回退：`HS_MUMPS=env tools/dev/build.sh release` 链接环境的 conda-forge `mumps-seq`
（P9 之前的构建）；`HS_MUMPS=/abs/prefix` 链接 `tools/dev/build_mumps.sh` 的其它安装；运行时
`HS_MUMPS_BLOCKING=mumps` 恢复 MUMPS 自己的块参数。`hsolve --A x --version` 打印所用的 MUMPS（`mumps=…`）。

构建出的 `hsolve` 默认链入 BLAS hook（`src/blas`，CMake `HSOLVE_BLAS_HOOK=ON`、route `all`）：MUMPS 回代中的窄
BLAS 调用改用自带的 AVX-512 / AVX2 窄核、L0 层之上的大调用并行切分，其余原样交给 OpenBLAS（指令集运行时检测）。
运行时 `HS_BLAS_ROUTE=off` 关闭（纯转发），构建时 `-DHSOLVE_BLAS_HOOK=OFF` 去掉；已有构建目录要
`-UHSOLVE_BLAS_HOOK -UHSOLVE_BLAS_ROUTE_DEFAULT` 重新配置才取新默认值。Python 模块不带 hook，可预加载
`build/release/libhsolve_blashook.so`（`docs/PYTHON.md` §1，`docs/BLAS.md`）。

求解（结果格式见 `docs/FORMATS.md`）：

```bash
./build/release/hsolve --A A.npz [--B B.npz] --sigma 0.5 --k 10 --out out/      # σ 附近最近 10 个
./build/release/hsolve --A A.npz --B B.npz --config config.json --out out/       # 完整配置
./build/release/hsolve --A A.npz --interval 1.0 1.6 --out out/                   # 区间内全部（谱切片）
./build/release/hsolve --A A.npz [--B B.npz] --smallest 10 --out out/            # 代数最小 10 个（--largest 同理）
```

常用覆盖参数（覆盖 config.json 中的同名键，见 `docs/FORMATS.md` §2）：`--tol`、`--threads`、
`--memory-budget-gb GiB|auto`（默认 auto：由 cgroup / `HSOLVE_MEMORY_LIMIT_GB` / ulimit / 物理内存中最小的上限推出，
约 0.85 倍）、`--precision double|single|auto`（单精度只为省内存，未达标时自动以双精度重解）、`--slice-size N`（每个位移至多 N 个特征值，0 = 自动）、
`--block-size N`（Krylov 块宽，0 = 自动：HEP/GHEP 的 nearest 运行由块宽规则按 σ 处分解的因子条目等在
1/2/4/8 中选、需求 ≥ 8 对时至少 4、8 须比 4 有 10% 的模型收益，切片需求 ≥ 8 对时 4；GHIEP n ≥ 2e4 且
≥ 4 对时 2；1 = 单向量）、
`--zero-filter-tol TOL`（|λ| ≤ TOL·scale 视为非物理零模，按惯性排除；< 0 关闭）、
`--certificate exact|randomized|auto|off`（nearest 的完备性检验：exact = 双精度惯性计数（默认）；randomized =
在已有分解上的随机排除检验，失败概率 ≤ `certificate_delta`（默认 1e-12），只在 B 的条件数有证明时（B = I、
对角或对角占优）出结论，否则改用计数；auto = 按实测时间选能认证的最便宜者，含单精度计数（带 10 倍余量）；
off = 不检验）、`--log FILE`。
只在 config.json 中的键：`"inertia_counter": "auto"|"on"|"off"`（证书计数是否用只计惯性的
InertiaCounter；auto 由代价模型决定）、`"certificate_delta"`、`"ordering"`（−1 = 自适应（默认，16 线程代价
模型在 AMD / 自研嵌套剖分 / SCOTCH / METIS / PORD 中选，P9 起自研嵌套剖分带流精修、先于 SCOTCH）；0、2..7 = MUMPS ICNTL(7)；10 = 自研多线程嵌套剖分）。

输出 `eigenvalues.npy`、`eigenvectors.npy`（n×m，Fortran 序）、`report.json`（残差、完备性证书（方法、是否
严格、失败概率与文字陈述）、每片记录、分解统计、块宽、五个互斥阶段的计时（含证书 `time_certificate_s`）、
峰值内存、`solver` / `version`），
退出码 0 = 全部达标，1 = 未达标，2 = 错误。B 不定（GHIEP）时只支持「σ 附近最近 k 个」。

Python（构建时找到 pybind11 即生成 `build/release/python/hsolve`，详见 `docs/PYTHON.md`）：

```python
import hsolve                                   # PYTHONPATH=build/release/python
res = hsolve.solve(A, B, sigma=0.5, k=10)       # A、B：scipy 稀疏 / numpy 矩阵或 .npz/.npy/.mtx 路径
res.eigenvalues, res.eigenvectors, res.report["success"]
```

对照基准（SLEPc，独立 conda 环境 `os2026-ref`；`tools/bench/tune_slepc.py` 为每个测例与核数调出最快的
有效配置，存于 `tools/bench/slepc_profiles.json`，方法见 `results/baseline_tuning.md`）与独立评测：

```bash
python3 tools/run/guard.py --exclusive --timeout 2700 -- python3 tools/bench/compare.py --tag mytag \
    --cores 16-31 --slepc-profile best --reps 5 case1 case2 ...    # 结果在 results/mytag.md（--out-dir 可改）
python3 tools/run/guard.py -- python -m tools.eval.sweep results/raw/mytag      # 评测汇总 results/eval_mytag.md
```

compare.py 对每次运行都用 `tools/eval` 判定有效性，只在双方每次重复都有效时给加速比，因此须在 guard 下运行；
计时时外层 guard 持有测量槽（`--exclusive`，每次 ≤ 2700 s，长的对比分几次），compare.py 自己不加 `--exclusive`。
`--solvers`（逗号分隔、交错运行；默认 `hsolve,hsolve-nocert,slepc`）可选 `hsolve`（默认配置，certificate = exact）、
`hsolve-auto`（certificate = auto）、`hsolve-nocert`（关闭证书）与 `slepc`，未知名称在任何运行之前被拒绝。

所有运行都经过资源看门狗（共享机器；集群节点上的两条道见下）：

```bash
python3 tools/run/guard.py --exclusive --timeout 2700 -- ./build/release/hsolve ...      # 计时：测量槽
python3 tools/run/guard.py --skip-busy-check --timeout 600 -- ./build/release/hsolve ... # 冒烟：构建/测试道
```

## 集群节点（gpu01 / gpu02，空闲时 c001–c005）

节点为 CentOS 7、无外网，环境由本机 conda 包缓存离线重建（详见脚本头注释）：

```bash
tools/dev/deploy_cluster.sh gpu01                       # 环境、代码（git remote "cluster"）、测例数据
tools/dev/cluster_run.sh gpu02 -- tools/dev/build.sh release          # 在节点上编译当前 worktree 的提交
tools/dev/cluster_run.sh gpu02 -- python3 tools/run/guard.py --exclusive --timeout 900 -- ...  # 在测量槽里计时运行
tools/dev/cluster_run.sh gpu02 --wip -- tools/dev/test.sh release     # 连同未提交的已跟踪改动一起传过去
```

单次运行的规格与评测机一致：16 核、32 GiB。每个节点有一个**测量槽**（NUMA node 1 的 16 核 `16-31`，
同一时刻只允许一个运行，用 `guard.py --exclusive --timeout <s>` 占用，`guard.py --status` 查看占用情况）
和一条**构建/测试道**（`8-15`，`HS_JOBS=4`，单次 ≤ 12 GiB），后者用于编译、测试与冒烟运行，计时只在
测量槽里做。规则由 guard 强制执行，详见 `docs/DESIGN.md` §6。`tools/dev/cluster_nodes.sh [--pick]` 列出各节点是否空闲
（c001–c005 与其他用户共用，只在空闲时用）。SLEPc 环境的离线重建细节见
`docs/DEPENDENCIES.md`。节点上的默认 MUMPS 缓存在共享的 `$HOME/deps/mumps-cache`（`env/hosts/cluster-node.sh`），
任何克隆的第一次构建编译一次，之后所有克隆复用；构建链接其路径，使用中的安装不要删除。

## 目录

| 路径 | 内容 |
|---|---|
| `include/hsolve/` | 公共接口头文件（模块契约） |
| `src/{io,sparse,linalg,factor,ops,eigen,verify,driver,util}/` | 各模块实现（`verify`：用回代代替分解的完备性检验，由 driver 的 `certificate` 模式调用） |
| `src/blas/` | BLAS hook（不属于 libhsolve；默认链入 `hsolve`，或 `libhsolve_blashook.so` 预加载） |
| `apps/` | 命令行程序 `hsolve` |
| `python/` | Python 接口（pybind11 扩展 `hsolve._hsolve` + `hsolve` 包；`python/bench` 开销测量） |
| `tests/cpp`, `tests/py` | GoogleTest 单元测试；Python 工具与接口测试（pytest） |
| `tools/gen` | 测例生成（波导有限元、Yee 网格、合成谱、SuiteSparse 下载） |
| `tools/eval` | 结果评测（残差、误差、子空间夹角、完备性）、批量汇总 `sweep.py` 与 scipy 基线 |
| `tools/bench` | SLEPc 基线、调优器 `tune_slepc.py`、`-log_view` 解析与对比脚本 `compare.py` |
| `bench/` | 可复现的 C++ 性能/内存测量程序（`bench_<name>`） |
| `tools/run/guard.py` | 资源看门狗（绑核、内存上限、互斥） |
| `tools/dev` | 构建（含默认 MUMPS 的源码构建 `build_mumps.sh` / `mumps_cache.sh`）、测试、集群部署与集群运行脚本 |
| `tools/deps` | 实验用 BLAS 替换库（OpenBLAS / BLIS / LIBXSMM）的源码构建脚本 |
| `third_party/mumps`, `patches/` | MUMPS 5.8.2 官方源码包（CeCILL-C）与 hsolve 对它的补丁（默认构建使用） |
| `env/` | conda 环境、激活脚本与按主机的设置 `env/hosts/` |
| `results/` | 实验汇总（原始输出在 `results/raw/`，不入库） |
