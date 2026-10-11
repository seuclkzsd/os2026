# hsolve 设计文档

hsolve 是面向电磁仿真的大规模稀疏厄密（含非正定、广义不定）特征值求解器，服务于
「中国商飞赛题：非正定厄密稀疏矩阵特征值求解」。本文件是**各模块开发的契约**：接口以
`include/hsolve/*.hpp` 为准，算法与约定以本文为准；二者冲突时先改文档再改代码。

赛题原文与宣讲材料整理见仓库根目录 `商飞赛题_非正定厄密稀疏矩阵特征值求解.md`。

---

## 1. 问题范围

| 类型 | 形式 | 说明 |
|---|---|---|
| HEP | A x = λ x | A 厄密（实对称或复厄密），通常不定 |
| GHEP | A x = λ B x | A 厄密，B 厄密正定（质量矩阵） |
| GHIEP | A x = λ B x | A、B 厄密，B 不定。宣讲真实算例即此类（Lee–Sun–Cendes 波导公式，λ = −β²） |

- 规模：目标 n ≈ 1e5，测试覆盖到 1e6；nnz 可达 1e8 量级。
- 求解目标：σ 附近最近的 k 个；区间 [a, b] 内全部；代数最小 / 最大的 k 个。
- 硬约束（赛题红线）：**禁止把稀疏矩阵显式稠密化**；禁止硬编码答案、读取参考结果、针对
  公开用例调参。所有算法参数必须由矩阵性质与用户配置按文档化规则推导。
- 运行环境：单机多核 CPU（≥8 核），内存 16 / 32 GB；GPU 可选。

## 2. 总体架构（严格分层，依赖只能自上而下）

```
apps/hsolve_cli            命令行：读配置 → 调 driver → 写结果
python/ (后续)             pybind11 接口，直接接收 scipy 稀疏矩阵
        │
src/driver   策略选择、位移管理、惯性证书、精修、残差验证、报告(JSON)
        │
src/eigen    Krylov-Schur（厄密 = 厚重启 Lanczos）、Krylov-Schur（一般/非厄密）、[LOBPCG、JD：P3]
        │        ↑ 只通过 LinearOperator 抽象（include/hsolve/operator.hpp）访问矩阵
src/ops      LinearOperator 的实现：CsrOperator（A）、ShiftInvertOperator（(A−σB)⁻¹B）
        │
src/factor   稀疏直接法（MUMPS d/z/s/c）：分析、预估内存、分解、多右端回代、惯性
src/sparse   CsrMatrix 工具、SpMV/SpMM、ShiftedPencil(A−σB)、复厄密→实对称嵌入
src/linalg   DenseMatrix、BLAS/LAPACK 声明与薄封装、块正交化(CGS2)
src/io       npy / npz / mtx 读写、三角存储恢复、厄密性检查
src/util     日志(JSON lines)、计时、内存(RSS)
```

- 下层模块（io / sparse / linalg / factor / eigen）**不打印、不依赖日志**；统计信息通过返回
  的结构体（`FactorStats`、`*EigResult::history`）交给 driver，由 driver 统一记录。
- 头文件分两类：`include/hsolve/` 为公共接口；`src/<module>/*.hpp` 为模块内部头文件。

## 3. 模块规格

### 3.1 io（`include/hsolve/io.hpp`）
- `.npy`：版本 1.0/2.0/3.0 头；dtype 支持 `<f8 <f4 <c16 <c8 <i4 <i8 |S*`；C 与 Fortran 序。
- `.npz`：zip（stored / deflate，zlib 解压），支持 zip64；`scipy.sparse.save_npz` 的
  csr / csc / coo 三种格式（键 `format, shape, data, indices, indptr` 或 `row, col`）。
- `.mtx`：MatrixMarket coordinate，real / complex / integer / pattern，general / symmetric /
  hermitian / skew-symmetric（后者报错）。
- 稠密 `.npy` 二维数组允许作为输入（小矩阵），转换为 CSR 时丢弃精确零。
- 三角存储恢复：`Storage::Auto` 时，若严格下三角为空且严格上三角非空则视为上三角存储
  （下三角同理），恢复为 `U + Uᴴ − diag(U)`；若用户声明 Full 但检测到单侧为空，按 Full
  处理并在 `LoadedMatrix::notes` 里记录。
- 厄密性检查：`‖A − Aᴴ‖_F / ‖A‖_F`，超过 `hermitian_tol` 抛 `Error::Code::NotHermitian`。
- 输出：`write_npy` 写实/复向量与 Fortran 序矩阵（numpy 可直接 `np.load`）。

### 3.2 sparse（`include/hsolve/sparse.hpp`）
- `CsrMatrix<T>` 不变量：`rowptr` 单调、行内列索引严格递增、无重复项、无越界。
- `csr_from_coo`：合并重复项（求和），可选丢弃零。
- `ShiftedPencil<T>`：一次性计算 A ∪ B 的并模式与索引映射，之后 `assemble(σ)` 只做 O(nnz)
  数值填充，用于多位移重分解时复用符号分析。
- `spmv / spmm`：OpenMP 按行并行，`Y = αAX + βY`；P3 做 AVX-512 优化（结果须与参考实现
  在 1e-14 相对误差内一致）。
- `real_embedding`：复厄密 K → 2n 阶实对称 `[[Re K, −Im K],[Im K, Re K]]`，其特征值为 K 的
  每个特征值各两份，惯性为 K 的两倍（MUMPS 不支持复厄密 LDLᴴ，见 3.4）。

### 3.3 linalg（`include/hsolve/dense.hpp`、`src/linalg/`）
- `DenseMatrix<T>`：列主序、64 字节对齐、`ld == rows`。
- `src/linalg/blas_lapack.hpp`：自行声明所需 Fortran BLAS/LAPACK 符号（不依赖 LAPACKE）。
- 正交化：块经典 Gram–Schmidt 两遍（CGS2），支持 B 内积（传入 BV 以避免重复乘 B）。

### 3.4 factor（`include/hsolve/factor.hpp`）
- 后端 MUMPS 5.8（顺序版 + OpenMP）。实数用 `SYM=2`（LDLᵀ，Bunch–Kaufman 型 2×2 主元）
  得到惯性 `INFOG(12)`；**复数 `SYM=2` 是复对称而非厄密**，因此复厄密矩阵只能用 LU
  (`SYM=0`)，惯性需走 `real_embedding`。工厂函数对「复数 + SymmetricIndefinite」直接报错。
- `analyze` 后给出 `FactorEstimate`（来自 INFOG(3)/(16)/(17)），driver 据此做内存预算。
- 单精度（smumps / cmumps）用于混合精度与内存受限场景；输入/输出仍为双精度，内部转换。
- 零主元检测 `ICNTL(24)=1`；奇异时抛 `Error::Code::Singular`，由 driver 扰动位移重试。
- 同一模式可多次 `factorize`（复用分析），用于多位移与惯性证书。

### 3.4b ops（`include/hsolve/operator.hpp`）
- `CsrOperator`：Y = A X（调用 `spmm`）。
- `ShiftInvertOperator`：Y = K⁻¹ B X —— 先 `spmm` 得 B X（B = I 时复制），再一次多右端
  `solve`（块回代，BLAS3）。计数 `applications()` 按向量数累计。

### 3.5 eigen（`include/hsolve/eigen.hpp`）
- `lanczos_krylov_schur`：要求算子在内积 ⟨x,y⟩_B = yᴴBx 下自伴（B=I 时即厄密）。位移反演
  OP = (A−σB)⁻¹B 在 B 正定时满足该条件。全正交化（CGS2，含已锁定向量），厚重启，
  收敛向量锁定，随机初始向量由 seed 决定，可传入 `deflation`（已知向量，求解时与之 B 正交）。
- `krylov_schur_general`：非厄密 Krylov-Schur（实数用实 Schur 形 + `dtrsen` 重排，不拆开 2×2
  块；复数用 `ztrsen`），用于 GHIEP（B 不定）的 OP = (A−σB)⁻¹B。返回 Schur 基与特征向量。
- 两者都通过 `history` 返回每次重启的收敛数与最大残差估计，用于可观测性与收敛分析。

### 3.6 driver（`include/hsolve/solver.hpp`）
流程：
1. **诊断**：规模、nnz、实/复、厄密性、B 是否存在、B 是否正定（`BDefinite::Auto` 时对 B 做
   一次 LDLᵀ 看惯性；复数用实嵌入）、矩阵范数估计。
2. **策略**：HEP/GHEP → 位移反演 + `lanczos_krylov_schur`（B 内积）；GHIEP →
   位移反演 + `krylov_schur_general`。两端特征值且 k 小 → 不分解，直接 Lanczos。
3. **内存预算**：`analyze` 后用估计值与 `memory_budget_gb` 比较：超预算依次尝试 BLR 压缩、
   单精度分解、out-of-core；仍超预算则以 `OutOfMemoryBudget` 失败并给出估计值（绝不盲目分解）。
4. **近奇异位移**：分解报奇异 / 检出零主元时，σ ← σ + δ（δ 从 1e-8·scale 起每次 ×10，最多 5 次），
   记录到报告；「最近 k 个」的选择始终相对原始 σ。
5. **迭代求解**：内层容差 = `tol × inner_tol_factor`；θ → λ = σ + 1/θ。
6. **精修**：对 HEP/GHEP 在原问题上做 Rayleigh–Ritz（小规模稠密 `zhegv/dsygv`），保证 B 正交；
   GHIEP 逐向量取 Rayleigh 商 λ = xᴴAx / xᴴBx（分母过小则保留 θ 映射值并告警）。
7. **残差验证**：对每个特征对计算 `‖Ax − λBx‖₂`、相对残差与后向误差（定义见 4.5），
   未达 `tol` 的对：用现有分解做 Rayleigh 商迭代补救；仍不达标则标记 `converged=false`。
8. **惯性证书**（HEP/GHEP）：在求得的窗口两侧的谱隙中点取 a、b，计算 ν(A−bB) − ν(A−aB)
   与找到的个数比较；不足说明漏根 → 以已得向量为 deflation 重跑补齐；结果写入报告。
9. **零模过滤**（可选）：|λ| ≤ `zero_filter_tol`·scale 的特征对视为非物理零模剔除并补算。
10. **输出**：`eigenvalues.npy`、`eigenvectors.npy`（n×k，Fortran 序）、`report.json`。

### 3.7 util（`include/hsolve/log.hpp`、`include/hsolve/util.hpp`）
- `Logger`：JSON lines 写文件 + 人类可读 stderr；事件名小写下划线，字段扁平。
- `Timer`、`peak_rss_bytes()`、`current_rss_bytes()`。

## 4. 数值约定

### 4.1 位移反演
K(σ) = A − σB，OP = K⁻¹B。A x = λ B x ⇔ OP x = θ x，θ = 1/(λ − σ)。离 σ 越近 |θ| 越大，
因此求 OP 的模最大特征值。B = I 时 OP = K⁻¹。

### 4.2 惯性（Sylvester）
B 正定时，ν₋(A − sB) = 「小于 s 的特征值个数」。用 LDLᵀ 的负主元数得到。区间 (a, b) 内
特征值数 = ν₋(A − bB) − ν₋(A − aB)。端点须远离特征值（取谱隙中点），否则数值惯性不可靠。

### 4.3 GHIEP
B 不定时惯性计数不成立，理论上可有复特征值。用非厄密 Krylov-Schur，结果中
|Im λ| > `complex_tol`·|λ| 的对按复数输出并告警；其余视为实特征值。

### 4.4 收敛（内层）
厄密 Krylov-Schur：Ritz 估计 |β_m s_{m,i}| ≤ tol_inner·|θ_i|。一般 Krylov-Schur 同理（Schur 形
下的残差估计）。

### 4.5 残差与精度指标（报告中全部给出）
- 绝对残差 r = ‖A x − λ B x‖₂
- 相对残差 r / (|λ|·‖B x‖₂)（λ = 0 时退化为 r / ‖Bx‖₂）
- 后向误差 η = r / ((‖A‖₁ + |λ|·‖B‖₁)·‖x‖₂)，**driver 以 η ≤ tol 作为接受标准**（‖·‖₁ 为
  2-范数的廉价上界代理，文档化）
- 与参考解比较时：特征值相对误差；特征向量用子空间夹角（重特征值按整个子空间比较）

## 5. 编码规范
- C++17，命名空间 `hs`；文件 `snake_case`；类型 `PascalCase`；函数/变量 `snake_case`；
  私有成员尾随下划线；常量 `kCamelCase`。格式见 `.clang-format`。
- 稠密矩阵列主序；全局索引 `index_t`（int64）；CSR 列索引 `int32_t`。
- 错误：抛 `hs::Error(code, message)`；输入错误不得用 assert。内部不变量可用 `HS_ASSERT`。
- 每个公共函数在头文件注明：前置条件、复杂度、线程安全性、是否修改输入。
- **禁止**：稀疏矩阵稠密化（唯一例外：测试代码中 n ≤ 2000 的参考解）；全局可变状态；
  无种子随机数；库代码中 `std::cout`/`printf`。
- 测试：`tests/cpp/test_<module>*.cpp`（GoogleTest），数值测试尽量使用有解析解的问题；
  Python 工具测试在 `tests/py/`（pytest）。
- 新增依赖必须是开源许可，并登记到 `docs/DEPENDENCIES.md`。

## 6. 资源与实验规范
本机为共享服务器（2×Xeon Gold 6336Y，48 物理核，125 GB）。**所有编译之外的运行都必须
经过 `tools/run/guard.py`**：
- 默认绑定 NUMA node1 的 16 个物理核 `24-30,33-41`，进程树 RSS 上限 30 GiB，系统可用内存
  低于 12 GiB 立即终止。
- 重型实验加 `--exclusive`（全机串行）；单元测试用 `--skip-busy-check --mem-gb 4`。
- 预估内存（MUMPS 分析）超过 28 GiB 的实验不运行，记录并上报。
- 编译并行度不超过 `-j4`。

## 7. 版本管理
- `main` 始终可编译、测试全过。功能在分支（或 worktree）开发，合并前须通过评审与测试。
- 提交信息：`<module>: <summary>`（如 `factor: add single-precision MUMPS backend`）。
- 生成数据（`data/`）、构建目录与原始实验输出不入库；实验汇总（`results/*.md|json`）入库。

## 8. 路线图
| 阶段 | 内容 |
|---|---|
| P0 | 仓库骨架、环境、资源看门狗、接口与设计文档 |
| P1 | io / sparse / linalg / factor / eigen 实现与单测；测例生成器；评测工具 |
| P2 | driver + CLI；端到端测试；测例集与首轮实验（对比 scipy 基线） |
| P3 | 性能剖析与优化（块回代、SpMV AVX-512、MUMPS 参数、混合精度）；LOBPCG / JD；区间切片 |
| P4 | 故障场景验证矩阵、方案文档、复现材料 |
