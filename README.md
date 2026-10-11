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
- 实测汇总：`results/p3_measurements.md`；测量程序：`bench/README.md`

## 快速开始

```bash
conda env create -f env/environment.yml      # 一次性
source env/activate.sh                       # 激活环境并设置资源策略默认值
tools/dev/build.sh release                   # 构建 (build/release)
tools/dev/test.sh release                    # C++ 单元测试 + Python 测试（工具与接口）
```

求解（结果格式见 `docs/FORMATS.md`）：

```bash
./build/release/hsolve --A A.npz [--B B.npz] --sigma 0.5 --k 10 --out out/      # σ 附近最近 10 个
./build/release/hsolve --A A.npz --B B.npz --config config.json --out out/       # 完整配置
./build/release/hsolve --A A.npz --interval 1.0 1.6 --out out/                   # 区间内全部（谱切片）
./build/release/hsolve --A A.npz [--B B.npz] --smallest 10 --out out/            # 代数最小 10 个（--largest 同理）
```

常用覆盖参数（覆盖 config.json 中的同名键，见 `docs/FORMATS.md` §2）：`--tol`、`--threads`、
`--memory-budget-gb`、`--precision double|single`、`--slice-size N`（每个位移至多 N 个特征值，0 = 自动）、
`--zero-filter-tol TOL`（|λ| ≤ TOL·scale 视为非物理零模，按惯性排除；< 0 关闭）、`--log FILE`。

输出 `eigenvalues.npy`、`eigenvectors.npy`（n×m，Fortran 序）、`report.json`（残差、完备性证书、每片记录、
分解统计、分阶段计时、峰值内存），退出码 0 = 全部达标，1 = 未达标，2 = 错误。B 不定（GHIEP）时只支持
「σ 附近最近 k 个」。

Python（构建时找到 pybind11 即生成 `build/release/python/hsolve`，详见 `docs/PYTHON.md`）：

```python
import hsolve                                   # PYTHONPATH=build/release/python
res = hsolve.solve(A, B, sigma=0.5, k=10)       # A、B：scipy 稀疏 / numpy 矩阵或 .npz/.npy/.mtx 路径
res.eigenvalues, res.eigenvectors, res.report["success"]
```

对照基准（SLEPc，独立 conda 环境 `os2026-ref`）与独立评测：

```bash
python tools/bench/compare.py --tag mytag --cores 24-30,33-41 case1 case2 ...   # 结果在 results/mytag.md
python3 tools/run/guard.py -- python -m tools.eval.sweep results/raw/mytag      # 评测汇总 results/eval_mytag.md
```

所有运行都经过资源看门狗（共享机器）：

```bash
python3 tools/run/guard.py --mem-gb 30 --timeout 3600 -- ./build/release/hsolve ...
```

## 目录

| 路径 | 内容 |
|---|---|
| `include/hsolve/` | 公共接口头文件（模块契约） |
| `src/{io,sparse,linalg,factor,ops,eigen,driver,util}/` | 各模块实现 |
| `apps/` | 命令行程序 `hsolve` |
| `python/` | Python 接口（pybind11 扩展 `hsolve._hsolve` + `hsolve` 包；`python/bench` 开销测量） |
| `tests/cpp`, `tests/py` | GoogleTest 单元测试；Python 工具与接口测试（pytest） |
| `tools/gen` | 测例生成（波导有限元、Yee 网格、合成谱、SuiteSparse 下载） |
| `tools/eval` | 结果评测（残差、误差、子空间夹角、完备性）、批量汇总 `sweep.py` 与 scipy 基线 |
| `tools/bench` | SLEPc 基线与对比脚本 `compare.py` |
| `bench/` | 可复现的 C++ 性能/内存测量程序（`bench_<name>`） |
| `tools/run/guard.py` | 资源看门狗（绑核、内存上限、互斥） |
| `env/` | conda 环境与激活脚本 |
| `results/` | 实验汇总（原始输出在 `results/raw/`，不入库） |
