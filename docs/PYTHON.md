# hsolve Python 接口

进程内 Python API：直接接收 numpy / scipy 矩阵，返回 numpy 特征值与特征向量（交付物 b
「numpy / scipy 矩阵输入，numpy / scipy 输出」）。实现：`python/src/module.cpp`（pybind11 扩展
`hsolve._hsolve`）、`python/src/csr_convert.hpp`（输入转换）、`python/hsolve/__init__.py`（纯
Python 前端）。求解本身就是 `hs::solve`——与命令行 `hsolve` 同一条路径、同一份配置格式。

## 1. 构建与导入

```bash
tools/dev/build.sh release                   # 找到 pybind11（环境内 3.1）时自动构建 Python 模块
PYTHONPATH=build/release/python python -c "import hsolve; print(hsolve.version())"
```

- 产物在 `build/<preset>/python/hsolve/`（`__init__.py` + `_hsolve.cpython-*.so`），无需安装。
- CMake 选项 `HSOLVE_BUILD_PYTHON`：找到 pybind11 时默认 ON；`-DHSOLVE_BUILD_PYTHON=OFF` 关闭，
  显式 ON 而缺 pybind11 时报错。debug 预设带 ASan/UBSan，普通 python 须预加载运行库才能导入：
  `LD_PRELOAD=$CONDA_PREFIX/lib/libasan.so.8:$CONDA_PREFIX/lib/libubsan.so.1 ASAN_OPTIONS=detect_leaks=0
  HSOLVE_PYTHON_DIR=build/debug/python python -m pytest tests/py/test_python_binding.py`
  （2026-10-11 实测 62 项全过、无 sanitizer 报告，约 5 分钟）；日常使用 release。
- 依赖：numpy；scipy 仅在传入 scipy 稀疏矩阵或调用 `load_matrix` 时需要。
- 测试：`python -m pytest tests/py/test_python_binding.py`（`tools/dev/test.sh release` 也会运行）；
  默认从 `build/release/python` 导入，`HSOLVE_PYTHON_DIR=<dir>` 指定其它构建；未构建时整个文件跳过。
- **BLAS hook**（P5，`docs/BLAS.md`、`docs/DESIGN.md` §3.8）：命令行 `hsolve` 默认链入了 hook（route all，
  16 线程端到端 1.05–1.16×，收益在回代），Python 扩展模块没有。要在 Python 中得到同样的路由，在解释器前
  预加载同一构建的 `libhsolve_blashook.so`（da4be7d 起可用）：

  ```bash
  LD_PRELOAD=$PWD/build/release/libhsolve_blashook.so HS_BLAS_ROUTE=all \
    PYTHONPATH=build/release/python python3 -c "import hsolve; r = hsolve.solve('A.npz', config='config.json'); print(r.report['success'])"
  ```

  扩展模块、MUMPS 与 OpenBLAS 在 Python 下由 `dlopen(RTLD_LOCAL)` 打开；预加载的 hook 仍排在它们的符号
  查找之前，并自己在这些已加载的库里找到真实的 BLAS 例程。同一进程中 numpy / scipy 的 BLAS 调用也经过
  hook（转发，或像 MUMPS 的调用一样改道；路由对任何调用方都正确）。`HS_BLAS_ROUTE` 不设时取构建默认值
  （`HSOLVE_BLAS_ROUTE_DEFAULT`，默认 all），`HS_BLAS_ROUTE=off` 为纯转发。只验证了正确性，Python 下的
  加速未单独计时。
- **MUMPS**（P9 集成，`docs/DESIGN.md` §3.4）：扩展模块与命令行链接同一个 MUMPS——默认构建（`HS_MUMPS=source`）
  是由仓库内 `third_party/mumps/` 源码包加 L0 分块补丁编译的 MUMPS 5.8.2，其 `lib/` 排在模块 RPATH 的最前面，
  先于环境的 conda-forge `mumps-seq`（库名与 soname 相同），默认块参数为 P9 的（运行时 `HS_MUMPS_BLOCKING=mumps`
  恢复 MUMPS 自己的）；`HS_MUMPS=env tools/dev/build.sh release` 改用环境的 MUMPS。`hsolve.build_info()` 写出
  所用的 MUMPS（`mumps=mumps-5.8.2-<key>` 或 `mumps=env`）与默认块参数；`tests/py/test_mumps_build.py` 在全新
  解释器中求解后检查 `/proc/self/maps` 里映射的正是所配置的库。

## 2. 快速上手

```python
import numpy as np, scipy.sparse as sp, hsolve

# HEP：A x = λ x，σ 附近最近的 6 个
res = hsolve.solve(A, sigma=0.3, k=6)
res.eigenvalues            # float64 (m,)，按离 σ 由近到远
res.eigenvectors           # (n, m) Fortran 序；A 复数时 complex128
res.report["success"]      # 每对后向误差 ≤ tol 且（HEP/GHEP）惯性证书通过

# GHEP / GHIEP（B 不定）：同一个调用，问题类型由 B 的惯性自动判定
res = hsolve.solve(A, B, sigma=-2.5e5, k=10)
res.complex_eigenvalues    # 仅 B 不定且出现真复特征值时非空，complex128 (c,)
res.complex_eigenvectors   # complex128 (n, c)

# 其它目标
hsolve.solve(A, lower=0.5, upper=0.7)          # 区间内全部（mode 自动推断为 interval）
hsolve.solve(A, mode="smallest", k=4)          # 代数最小（标准问题）
hsolve.solve(A, mode="largest", k=4)

# 测例目录 / 配置文件：与 CLI 完全相同的读入流水线
res = hsolve.solve("case/A.npz", "case/B.npz", config="case/config.json")
A = hsolve.load_matrix("case/A.npz")           # scipy.sparse.csr_array，值与列索引不拷贝
```

## 3. API

### `hsolve.solve(A, B=None, *, sigma=None, k=6, mode='nearest', lower=None, upper=None, tol=1e-10, config=None, threads=0, verbose=0, storage='auto') -> SolveResult`

| 参数 | 含义 |
|---|---|
| `A`, `B` | scipy 稀疏矩阵/数组（csr、csc、coo 直接转换；bsr、dia、lil、dok 经 `.tocsr()`），稠密 numpy 数组（含 `np.matrix`、任意步长视图），或 `.npz/.npy/.mtx` 文件路径。`B=None` 即 B = I |
| `sigma` | nearest 模式的位移；给出且未给 `mode` 时 mode 取 `nearest` |
| `k` | 特征对个数（nearest / smallest / largest） |
| `mode` | `nearest` \| `interval` \| `smallest` \| `largest`；省略时：给了 `lower/upper` → `interval`，给了 `sigma` → `nearest`，否则取 config（默认 nearest）。同时给 `sigma` 与 `lower/upper` 而不给 `mode` 报错 |
| `lower`, `upper` | interval 模式的闭区间 |
| `tol` | 接受阈值：每对后向误差 η ≤ tol（`docs/DESIGN.md` §4.5） |
| `config` | dict 或 `config.json` 路径（`docs/FORMATS.md` §2 的全部键；未知键报错）。**显式传入**的关键字参数覆盖它，未传的关键字不覆盖 |
| `threads` | OpenMP 线程数，0 = `OMP_NUM_THREADS` / 全部可用；调用结束后恢复进程原线程设置 |
| `verbose` | 0 静默，1 进度事件，2 调试；写到进程的 stderr（C 层，Jupyter 中显示在终端） |
| `storage` | `auto`（某个严格三角为空时视为三角存储并镜像，与 CLI 相同）\| `full` \| `upper` \| `lower`（声明只存该三角，另一三角出现元素报错）；同时作用于 A 与 B |

`SolveResult` 字段：

| 字段 | 内容 |
|---|---|
| `eigenvalues` | float64 (m,)，实特征值，按目标排序（nearest：离 σ 由近到远；interval / smallest：升序；largest：降序） |
| `eigenvectors` | (n, m)，Fortran 序；A 或 B 为复数 dtype（内存数组的 dtype，或文件中存储的值类型）时 complex128——虚部全为零、按实数求解时也是——否则 float64；HEP/GHEP 为 B-正交归一，GHIEP 为 2-范数归一 |
| `complex_eigenvalues`, `complex_eigenvectors` | GHIEP 的真复特征对，complex128 (c,) 与 (n, c)；通常为空 |
| `report` | dict，与 CLI `report.json` 相同的键（`docs/FORMATS.md` §3）；输入预处理的说明以 `"input: A: ..."` 追加到 `report["warnings"]` |
| `config` | dict，本次生效的**完整**配置——存成 `config.json` 即可用 CLI 或 `solve(config=...)` 复现 |
| `success` | 属性，等于 `report["success"]` |

未达标（`success == False`）时发出 `hsolve.HsolveWarning`，并照常返回结果（与 CLI 退出码 1 对应）。

### 其它函数

- `hsolve.load_matrix(path, *, storage='auto', hermitian_tol=1e-10, return_info=False)`：用 C++ 读入器
  （与 CLI 相同：合并重复项、丢弃精确零、三角恢复、厄密检查与厄密化、零虚部复数降为实数）读文件，
  返回 `scipy.sparse.csr_array`：`data` 与 `indices`（int32）直接引用 C++ 缓冲区（不拷贝），`indptr`
  是 n+1 个元素的 int32 副本（C++ 行指针随即释放）。原因：scipy 的 csr_array 要求 indptr 与 indices
  同一种索引类型，只要有一个是 int64 就把另一个也拷成 int64；直接交出 int64 行指针会让它把全部列索引
  拷一遍，而 C++ 的 int32 原件仍被占用（每个非零元 12 B 而非 4 B）。n 或 nnz 超出 int32 时 indices 为
  一份 int64 副本（scipy 本来也要做这次拷贝；C++ 原件随即释放），indptr 引用 C++ 缓冲区。实测见 §8。
  `return_info=True` 另返回 `{source_format, storage, hermitian_defect, notes}`。
- `hsolve.default_config()`：全部配置键的默认值（dict）。
- `hsolve.version()` / `hsolve.__version__`、`hsolve.build_info()`。

### 异常

`hs::Error` 的错误码映射为 `HsolveError` 子类（属性 `code` 为错误码字符串），并同时继承最贴切的
Python 内置异常，`except ValueError` 等惯用写法照常生效：

| code | 类 | 也是 | 典型原因 |
|---|---|---|---|
| `invalid_input` | `InvalidInputError` | `ValueError` | k ≤ 0 或 k > n、非方阵、A/B 阶数不同、非有限值（含重复项求和溢出）、越界索引、未知配置键、非法 mode/storage |
| `io` | `HsolveIOError` | `OSError` | 矩阵或配置文件不存在/损坏 |
| `not_hermitian` | `NotHermitianError` | `ValueError` | ‖M−Mᴴ‖_F/‖M‖_F > `hermitian_tol` |
| `unsupported` | `UnsupportedError` | `NotImplementedError` | 当前版本不支持的组合（如 B 不定时的 interval） |
| `factorization_failed` | `FactorizationFailedError` | `RuntimeError` | MUMPS 失败 |
| `singular` | `SingularError` | `ArithmeticError` | σ 处及其附近 A−σB 都奇异 |
| `out_of_memory_budget` | `OutOfMemoryBudgetError` | `MemoryError` | 预测内存超 `memory_budget_gb`（未分配） |
| `not_converged` | `NotConvergedError` | `RuntimeError` | — |
| `internal` | `InternalError` | `RuntimeError` | 内部不变量被破坏（缺陷，请报告） |

参数类型错误（如 `k=2.5`、`sigma='0.5'`、`config=42`、传入 `LinearOperator`）抛 `TypeError`：实数参数
（`sigma/lower/upper/tol/hermitian_tol`）接受 int、float、numpy 整数/浮点标量与 0 维数组，整数参数
（`k/threads/verbose`）接受 int 与 numpy 整数，字符串一律不接受。
测试 `test_every_error_code_has_its_class` 用 C++ 枚举导出的码表核对上表 9 个类，并逐个检查翻译。

## 4. 输入：格式、类型与拷贝

每个矩阵**一次遍历**转换为求解器的 `hs::CsrMatrix`（拥有内存的 CSR：int64 行指针、int32 列索引），
dtype 转换在这次写入中完成；稀疏输入从不稠密化。

| 输入 | 转换 | 额外拷贝 |
|---|---|---|
| csr（行内已排序） | 逐行直接复制 | 1 次（写入 CsrMatrix） |
| csr（行内乱序/重复） | 仅这些行排序（稳定）+ 合并 | 1 次 |
| csc | **不转置**：csc 数组即 Aᵀ 的 CSR，取共轭得 Aᴴ；Aᴴ 与 A 的厄密部分 (A+Aᴴ)/2 与厄密缺陷完全相同，而求解器只用这两者（A 精确厄密时 Aᴴ 与 A 逐位相等） | 1 次 |
| coo | 按行计数排序直接写入 CSR 数组，再逐行排序 / 合并 | 1 次 |
| 稠密 ndarray（C/F 序或任意步长） | 计数一遍，复制非零元（精确零视为不存在，同 CLI 的 `.npy`） | 1 次（仅非零元） |
| bsr / dia / lil / dok | scipy `.tocsr()` 后同 csr | 2 次 |
| 文件路径 | `hs::load_matrix`（同 CLI） | 0 次（读入即 CsrMatrix） |

- 值 dtype：float32/64、complex64/128 直接读取（float32 → double 精确）；整数/布尔/float16/longdouble
  先由 numpy 转为 float64（longdouble 有舍入）；索引 int32/int64 直接读取，其它整型转 int64。
- 实/复分派：A 或 B 含非零虚部 → 复数求解（实的一方在转换中提升，不另拷贝）；复数 dtype 但虚部
  全为零 → 按实数求解（更快，与 CLI 读文件的行为一致），特征向量仍以 complex128 返回——文件路径
  输入同样如此（文件中存的是复数类型即可）。
- 形状在转换之前按调用者给出的方向检查（`A is not square (30x20)`、`B has order 31, A has 30`）：
  csc 在内部是 Aᴴ，三角输入先要镜像，但报错不会出现转置后的形状或内部函数名。
- 规范化与 `hs::load_matrix` 相同：重复项求和、求和后的精确零丢弃、inf/nan 报 `InvalidInputError`
  （有限的重复项求和溢出成 inf 也一样，load_matrix 同样在求和之后检查）；三角存储按 `storage` 恢复；
  之后由 `hs::solve` 做厄密检查：P5 起与 `hs::load_matrix` 同一规则——缺陷 ‖M−Mᴴ‖_F/‖M‖_F 用缩放平方和
  计算（任意量级都准确），超 `hermitian_tol` 报 `NotHermitianError`，按逐元素精确比较不是精确厄密时取
  厄密部分；全零矩阵与非有限值报 `InvalidInputError`（`docs/DESIGN.md` §3.6-1）。P4 时内存输入用未缩放
  的平方和，元素模超出约 1e±154 时与路径输入结果不同，这一差异已消除。
- 内存：调用期间 scipy 持有的输入与求解器的 CsrMatrix 并存（≈ 输入大小 + 1 份 CSR）；转换后的
  矩阵在结果返回 Python 前释放。

## 5. 输出

- `eigenvectors` / `complex_eigenvectors` **不拷贝**：numpy 数组直接拥有 C++ 的特征向量缓冲区
  （`v.base` 是一个 PyCapsule，数组可写，生命周期由 Python 管理）。
- 特征值是长度 m 的小数组（复制）。`report`/`config` 由 JSON 解析为 dict。

## 6. GIL 与线程

- 转换与求解期间**释放 GIL**：其它 Python 线程照常运行（测试 `test_gil_is_released_during_solve`）。
- 同一进程内的多个 `solve` 调用**串行执行**（进程级互斥锁，在释放 GIL 之后获取）：单次求解已用满
  所配置的核，且 MUMPS 顺序版（libseq）未声明多实例并发安全。`load_matrix` 可并发。
- `threads>0` 只作用于本次调用，之后恢复进程原来的 OpenMP 线程数。
- 求解中按 Ctrl-C：信号在本次求解返回后才生效。
- 在调用期间不要修改传入的数组（与 numpy 释放 GIL 的其它函数相同的约定）。

## 7. 与命令行的一致性

- 配置：同一 schema、同一校验器（`hs::parse_config`）；`result.config` 可直接存为 `config.json`。
- 路径输入走与 CLI 相同的 `hs::load_matrix`（`hermitian_tol` 取自配置）；内存输入按同样的规则
  规范化，厄密检查也相同（§4）。测试（`tests/py/test_python_binding.py`）
  中 22 种实矩阵输入（csr/csc/coo 的 matrix 与 array 版本、int64 索引、乱序 + 重复项、显式零、
  bsr/lil、稠密 C/F 序与步长视图、三角存储……）与 9 种复矩阵输入在 `deterministic=true` 下得到
  **逐位相同**的特征值与特征向量。

## 8. 实测开销

方法可复现：`bench/pybind_convert.cpp`（转换内核）、`python/bench/overhead.py`（端到端），均经
`tools/run/guard.py --cores 38,39`（2 核，2026-10-11，release 构建）。

转换内核（`build/release/bench_pybind_convert 100 5`：三维 7 点 Laplacian，n = 1e6，nnz = 6.94e6，
中位数）：「下限」是任何拥有内存的 CsrMatrix 都要付的代价（分配 + 置零 + memcpy 索引/值 + 行指针加宽）。

| 输入 | 时间 (s) | 相对下限 |
|---|---|---|
| 下限（memcpy） | 0.036 | 1.00 |
| csr int32/f64 | 0.048 | 1.34 |
| csc int32/f64（不转置） | 0.049 | 1.36 |
| csr int64/f64 | 0.050 | 1.38 |
| csr int32/f32 | 0.049 | 1.36 |
| csr complex128（虚部全零：扫描 + 转实数） | 0.063 | 1.77 |
| csr 每行逆序（逐行排序） | 0.105 | 2.94 |
| coo 行有序 | 0.064 | 1.79 |
| coo 随机打乱 | 0.290 | 8.09 |
| 参照：库函数 `hs::csr_from_coo`（同一打乱三元组，int64） | 0.257 | 7.16 |
| csr 上三角 + 镜像 | 0.172 | 4.80 |

负结果（如实记录）：打乱的 coo 比 `hs::csr_from_coo` 慢 13%。分段计时表明随机散射占主导
（写两个数组 0.165 s，写成 (列, 值) 对数组 0.100 s）；改用对数组可省约 65 ms，但峰值内存每个非零元
多 16 B，因此保留省内存的写法。逐行排序把 `std::stable_sort`（每次调用分配临时缓冲）换成插入排序：
0.118 → 0.105 s（逆序行），coo 打乱 0.317 → 0.290 s。

端到端（`python/bench/overhead.py CASE --mode memory|path`，与 CLI 同 guard 设置；「开销」=
`solve()` 墙钟 − 报告中求解器自身的 `time_total_s`，两次重复的范围）：

| 测例 | n / nnz(A) | CLI 墙钟 / 求解 (s) | 内存输入开销 (s) | 路径输入开销 (s) | 峰值 GiB：CLI / 内存 / 路径 |
|---|---|---|---|---|---|
| cav2d-rect-pert-1e5（GHEP） | 99459 / 0.50M | 0.97 / 0.82 | 0.006–0.013 | 0.056–0.066 | 0.12 / 0.18 / 0.17 |
| wg-square-1e5（GHIEP） | 99225 / 0.37M | 1.97 / 1.83 | 0.010–0.019 | 0.049–0.059 | 0.28 / 0.36 / 0.34 |
| syn-near-singular-complex-1e5（复 HEP） | 100000 / 2.40M | 5.08 / 4.67 | 0.032–0.041 | 0.316–0.344 | 0.40 / 0.50 / 0.46 |
| yee-pec-box-1e5（GHEP） | 109744 / 1.39M | 9.34 / 9.16 | 0.038–0.046 | 0.177–0.191 | 0.71 / 0.78 / 0.76 |

- 内存输入：转换 + 结果交接占求解时间的 0.4–1.6%。路径输入的开销是读 npz（CLI 同样要付）。
- 峰值差：Python 解释器 + numpy/scipy 约 0.07 GiB（`import scipy.sparse, hsolve` 实测），内存输入另含
  scipy 持有的那份矩阵。两次重复的特征值差 ≤ 1e-13（默认非确定性模式，见 `deterministic`）。
- 重复项求和溢出检查（只在有重复项的列上多一次 `isfinite`）不改变转换耗时：上表的
  `bench_pybind_convert 100 5` 与修改前的二进制交替各跑 3 次（2026-10-11），各项中位数之比 0.92–1.09，
  在本机此时 ±10% 的噪声内（逐行排序 1.00，coo 打乱 0.92）。

`load_matrix` 返回的矩阵所占内存（`python/bench/load_memory.py PATH --mode hsolve|scipy`，每种各一个
进程，经 guard，2 核，2026-10-11；读入并 gc 后 RSS 的增长，两次重复；「改前」为 int64 indptr 版本）：

| A.npz | n / nnz | 改前 (MiB) | 现在 (MiB) | `scipy.sparse.load_npz` (MiB) | 三个数组本身 (MiB) |
|---|---|---|---|---|---|
| yee-pec-box-1e5（实） | 109744 / 1.39M | 30.7, 30.7 | 17.0, 18.2 | 18.7, 18.6 | 16.3 |
| syn-near-singular-complex-1e5（复） | 100000 / 2.40M | 66.8, 68.2 | 46.9, 46.9 | 47.1, 47.0 | 46.2 |
| wg-square-1e6（实） | 998001 / 3.74M | 80.6, 80.6 | 48.3, 47.2 | 48.4, 49.5 | 46.6 |

现在与 scipy 自己读文件持平。改前列索引占 12 B/nnz（scipy 拷成的 int64 副本 8 B，加仍被占用的 int32
原件 4 B），现在 4 B/nnz；行指针由每行 8 B 降为 4 B——合计与实测差值一致（如 wg-square-1e6：
3.74M × 8 B + 1e6 × 4 B ≈ 32 MiB）。读入耗时不变（0.06 / 0.37 / 0.15 s）。

## 9. 限制

- 需要显式矩阵（要分解），不接受 `LinearOperator`。
- 目标模式的支持范围随 driver：B 不定时只支持 nearest；interval / smallest / largest 要求 B 正定或 B = I。
- 完备性检验的方法（`certificate`）只能经 `config` 设定（无对应关键字参数），见 `docs/FORMATS.md` §2。
- 不同线程中的求解串行执行（见 §6）。
