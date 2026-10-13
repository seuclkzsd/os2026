# hsolve — 大规模非正定厄密稀疏矩阵特征值求解器

面向电磁仿真（微波器件、天线、电磁兼容、光子晶体、波导模式分析）的特征值求解器：
支持 `A x = λ x` 与 `A x = λ B x`（B 正定或**不定**），矩阵为实对称或复厄密、规模 ~1e5、
非正定；求目标位移 σ 附近 / 区间内 / 谱两端的特征对。核心为 C++17 + MUMPS（对称不定
LDLᵀ、惯性计数）+ 自研 Krylov-Schur（含块模式）、LOBPCG、Jacobi-Davidson；命令行与 Python 两种接口。

- 赛题：`商飞赛题_非正定厄密稀疏矩阵特征值求解.md`
- 设计与模块契约：`docs/DESIGN.md`
- 依赖与许可证：`docs/DEPENDENCIES.md`
- 方案说明（交付物 a）：`docs/SOLUTION.md`；文件格式：`docs/FORMATS.md`
- Python 接口（scipy/numpy 进出）：`docs/PYTHON.md`
- 实测汇总：`results/p3_measurements.md`、`results/p4_measurements.md`；SLEPc 调优与公平对比：
  `results/baseline_tuning.md`；测量程序：`bench/README.md`

## 快速开始

```bash
conda env create -f env/environment.yml      # 一次性
source env/activate.sh                       # 激活环境并设置资源策略默认值
tools/dev/build.sh release                   # 构建 (build/release)
tools/dev/test.sh release                    # C++ 单元测试 + Python 测试（工具与接口）
```

开发服务器自 P4 收尾起只做编辑与 git，编译、测试与实验都在集群节点上运行（见下「集群节点」）。

求解（结果格式见 `docs/FORMATS.md`）：

```bash
./build/release/hsolve --A A.npz [--B B.npz] --sigma 0.5 --k 10 --out out/      # σ 附近最近 10 个
./build/release/hsolve --A A.npz --B B.npz --config config.json --out out/       # 完整配置
./build/release/hsolve --A A.npz --interval 1.0 1.6 --out out/                   # 区间内全部（谱切片）
./build/release/hsolve --A A.npz [--B B.npz] --smallest 10 --out out/            # 代数最小 10 个（--largest 同理）
```

常用覆盖参数（覆盖 config.json 中的同名键，见 `docs/FORMATS.md` §2）：`--tol`、`--threads`、
`--memory-budget-gb`、`--precision double|single`、`--slice-size N`（每个位移至多 N 个特征值，0 = 自动）、
`--block-size N`（Krylov 块宽，0 = 自动：HEP/GHEP 需求 ≥ 8 对时 4、GHIEP n ≥ 2e4 且 ≥ 4 对时 2；1 = 单向量）、
`--zero-filter-tol TOL`（|λ| ≤ TOL·scale 视为非物理零模，按惯性排除；< 0 关闭）、`--log FILE`。
只在 config.json 中的 P4 新键：`"inertia_counter": "auto"|"on"|"off"`（证书计数是否用只计惯性的
InertiaCounter；auto 由代价模型决定）。

输出 `eigenvalues.npy`、`eigenvectors.npy`（n×m，Fortran 序）、`report.json`（残差、完备性证书、每片记录、
分解统计、块宽、五个互斥阶段的计时（含证书 `time_certificate_s`）、峰值内存、`solver` / `version`），
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
python3 tools/run/guard.py -- python tools/bench/compare.py --tag mytag --cores 24-30,33-41 \
    --slepc-profile best --reps 5 case1 case2 ...                                # 结果在 results/mytag.md（--out-dir 可改）
python3 tools/run/guard.py -- python -m tools.eval.sweep results/raw/mytag      # 评测汇总 results/eval_mytag.md
```

compare.py 对每次运行都用 `tools/eval` 判定有效性，只在双方每次重复都有效时给加速比，因此须在 guard 下运行。

所有运行都经过资源看门狗（共享机器）：

```bash
python3 tools/run/guard.py --mem-gb 30 --timeout 3600 -- ./build/release/hsolve ...
```

## 集群节点（gpu01 / gpu02 / c001）

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
测量槽里做。规则由 guard 强制执行，详见 `docs/DESIGN.md` §6。SLEPc 环境的离线重建细节见
`docs/DEPENDENCIES.md`。

## 目录

| 路径 | 内容 |
|---|---|
| `include/hsolve/` | 公共接口头文件（模块契约） |
| `src/{io,sparse,linalg,factor,ops,eigen,verify,driver,util}/` | 各模块实现（`verify`：用回代代替分解的完备性检验，尚未接入 driver） |
| `apps/` | 命令行程序 `hsolve` |
| `python/` | Python 接口（pybind11 扩展 `hsolve._hsolve` + `hsolve` 包；`python/bench` 开销测量） |
| `tests/cpp`, `tests/py` | GoogleTest 单元测试；Python 工具与接口测试（pytest） |
| `tools/gen` | 测例生成（波导有限元、Yee 网格、合成谱、SuiteSparse 下载） |
| `tools/eval` | 结果评测（残差、误差、子空间夹角、完备性）、批量汇总 `sweep.py` 与 scipy 基线 |
| `tools/bench` | SLEPc 基线、调优器 `tune_slepc.py`、`-log_view` 解析与对比脚本 `compare.py` |
| `bench/` | 可复现的 C++ 性能/内存测量程序（`bench_<name>`） |
| `tools/run/guard.py` | 资源看门狗（绑核、内存上限、互斥） |
| `tools/dev` | 构建、测试、集群部署与集群运行脚本 |
| `env/` | conda 环境、激活脚本与按主机的设置 `env/hosts/` |
| `results/` | 实验汇总（原始输出在 `results/raw/`，不入库） |
