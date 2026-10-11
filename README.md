# hsolve — 大规模非正定厄密稀疏矩阵特征值求解器

面向电磁仿真（微波器件、天线、电磁兼容、光子晶体、波导模式分析）的特征值求解器：
支持 `A x = λ x` 与 `A x = λ B x`（B 正定或**不定**），矩阵为实对称或复厄密、规模 ~1e5、
非正定；求目标位移 σ 附近 / 区间内 / 谱两端的特征对。核心为 C++17 + MUMPS（对称不定
LDLᵀ、惯性计数）+ 自研 Krylov-Schur。

- 赛题：`商飞赛题_非正定厄密稀疏矩阵特征值求解.md`
- 设计与模块契约：`docs/DESIGN.md`
- 依赖与许可证：`docs/DEPENDENCIES.md`
- 方案说明（交付物 a）：`docs/SOLUTION.md`；文件格式：`docs/FORMATS.md`

## 快速开始

```bash
conda env create -f env/environment.yml      # 一次性
source env/activate.sh                       # 激活环境并设置资源策略默认值
tools/dev/build.sh release                   # 构建 (build/release)
tools/dev/test.sh release                    # C++ 单元测试 + Python 工具测试
```

求解（结果格式见 `docs/FORMATS.md`）：

```bash
./build/release/hsolve --A A.npz [--B B.npz] --sigma 0.5 --k 10 --out out/      # σ 附近最近 10 个
./build/release/hsolve --A A.npz --B B.npz --config config.json --out out/       # 完整配置
./build/release/hsolve --A A.npz --interval 1.0 1.6 --out out/                   # 区间内全部
```

输出 `eigenvalues.npy`、`eigenvectors.npy`（n×m，Fortran 序）、`report.json`（残差、完备性证书、分解统计、
计时、峰值内存），退出码 0 = 全部达标，1 = 未达标，2 = 错误。

对照基准（SLEPc，独立 conda 环境 `os2026-ref`）：

```bash
python tools/bench/compare.py --tag mytag --cores 24-30,33-41 case1 case2 ...   # 结果在 results/mytag.md
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
| `tests/cpp`, `tests/py` | GoogleTest 单元测试；Python 工具测试 |
| `tools/gen` | 测例生成（波导有限元、Yee 网格、合成谱、SuiteSparse 下载） |
| `tools/eval` | 结果评测（残差、误差、子空间夹角）与 scipy 基线 |
| `tools/bench` | SLEPc 基线与对比脚本 `compare.py` |
| `bench/` | 可复现的 C++ 性能/内存测量程序（`bench_<name>`） |
| `tools/run/guard.py` | 资源看门狗（绑核、内存上限、互斥） |
| `env/` | conda 环境与激活脚本 |
| `results/` | 实验汇总（原始输出在 `results/raw/`，不入库） |
