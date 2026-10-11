# hsolve 设计文档

hsolve 是面向电磁仿真的大规模稀疏厄密（含非正定、广义不定）特征值求解器，服务于
「中国商飞赛题：非正定厄密稀疏矩阵特征值求解」。本文件是**各模块开发的契约**：接口以
`include/hsolve/*.hpp` 为准，算法与约定以本文为准；二者冲突时先改文档再改代码。
文件格式见 `docs/FORMATS.md`；赛题见根目录 `商飞赛题_非正定厄密稀疏矩阵特征值求解.md`。

修订记录：v2（P0 设计评审后）——采纳三路评审（数值 / 性能 / 可维护性）30 条意见，主要变化：
MUMPS 奇异与惯性约定实测化；铅笔（pencil）只存上三角索引并直接写入后端数组；复厄密统一走
实对称嵌入（求解 + 惯性一条路径）；单精度/BLR 必须配双精度迭代修正；证书窗口按目标类型定义；
后向误差改用 2-范数估计；GHIEP 实/复分类按尺度 + 残差；零模用移位放置而非剔除重算；
存储 B·V 改为可选；文件格式与参考解隔离；测试命名、实例化、所有权规则。

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
apps/hsolve_cli            命令行：读配置 → 调 driver → 写结果（docs/FORMATS.md §3）
python/ (后续)             pybind11 接口，直接接收 scipy 稀疏矩阵
        │
src/driver   诊断、策略、位移管理、惯性证书、精修、残差验证、报告(JSON)
        │
src/eigen    Krylov-Schur（厄密 = 厚重启 Lanczos）、Krylov-Schur（一般/非厄密）、[LOBPCG、JD：P3]
        │        ↑ 只依赖 include/hsolve/linear_operator.hpp 与 src/linalg
src/ops      LinearOperator 实现：CsrOperator（A）、ShiftInvertOperator（(A−σB)⁻¹B，可选迭代修正）
        │
src/factor   MUMPS：分析/估计/分解/多右端回代/惯性；复数走实嵌入
src/sparse   CsrMatrix 工具、SpMV/SpMM、ShiftedPencil（上三角并模式）、实嵌入、厄密化
src/linalg   DenseMatrix、BLAS/LAPACK 封装、随机向量、CGS2 正交化（dense_ops）
src/io       npy / npz / mtx 读写、三角存储恢复、厄密性检查与厄密化
src/util     错误、日志(JSON lines)、计时、内存(RSS)
```

- 下层模块（io / sparse / linalg / factor / ops / eigen）**不打印、不依赖日志**；统计信息通过
  返回的结构体（`FactorStats`、`*EigResult::history`）交给 driver 统一记录。
- 头文件：`include/hsolve/` 为公共接口；`src/<module>/*.hpp` 为模块内部头文件。

## 3. 模块规格

### 3.1 io（`include/hsolve/io.hpp`）
- `.npy`：版本 1.0/2.0/3.0；`parse_npy` 接受任意定长标量 descr（`<f8 <c16 <i4 <i8 |b1 |u1 <u8
  |S3 <U3`…，含大端），保留原始字节；类型转换只在 `npy_to_vector` 中进行（拒绝不支持的转换）。
  结构化/对象 dtype 拒绝。
- `.npz`：zip（stored / deflate，zlib），**成员大小与偏移取自中央目录及其 zip64 扩展字段**
  （numpy 以 force_zip64 写成员，本地头里是 0xFFFFFFFF），校验 CRC-32。
- scipy `save_npz`：csr / csc / coo，matrix 与 array 两种 API；忽略格式不用的键（如 `_is_array`
  `|b1`）；`format` 键可能是 `|S3` 或 `<U3`；索引 int32/int64；bsr / dia 抛 Unsupported。
- `.mtx`：coordinate（real/complex/integer/pattern × general/symmetric/hermitian；skew 报错）与
  array 格式。
- 稠密 `.npy` 二维数组允许作为输入（小矩阵），转换为 CSR 时丢弃精确零。
- 加载流水线：建 CSR（合并重复项）→ 三角存储恢复（`Storage::Auto`：严格下三角空且严格上三角
  非空视为上三角存储，反之亦然；用户声明 Upper 却出现下三角元素则报错）→ 厄密检查（超
  `hermitian_tol` 抛 NotHermitian）→ **缺陷 > 0 时以 `hermitian_part` 替换**并记入 notes（保证
  分解只读一个三角、算子与残差用全矩阵时看到的是同一个矩阵）→ 全零虚部的复数输入降为实数。
- 输出：`write_npy_vector`、`write_npy_matrix`（Fortran 序，np.load 得 (rows, cols)）。

### 3.2 sparse（`include/hsolve/sparse.hpp`）
- `CsrMatrix<T>` 不变量见头文件；列索引 int32（n ≤ 2³¹），行指针 int64。
- `csr_from_coo / conj_transpose / add / identity / triangle / symmetrize_from_triangle /
  hermitian_part / to_complex / norm_fro / norm_one / hermitian_defect / spmv / spmm /
  real_embedding`。
- `ShiftedPencil<T>`：A ∪ B 的**上三角并模式（只存索引，对角线必在每行首位）** + A、B 上三角
  元素到该模式的散射映射（upper(A) 与并模式逐行相同时 A 用恒等映射省内存）。
  `assemble(α, β, U* vals)` 直接把 αA + βB 写入后端的值数组（U 可为单精度），全量 K 从不物化。
  K(σ) = 1·A + (−σ)·B，B 自身 = 0·A + 1·B，同一模式服务所有位移与 B 正定性检测。
  `assemble_full` 仅供测试/小问题。
- spmv/spmm 为参考实现（OpenMP）；P3 的 SIMD 版本须与之在 1e-14 相对误差内一致。

### 3.3 linalg（`include/hsolve/dense.hpp`、`src/linalg/`）
- `DenseMatrix<T>`：列主序、64 字节对齐、`ld == rows`。
- `blas_lapack.hpp`：自行声明 Fortran BLAS/LAPACK（LP64，隐藏字符串长度参数显式传递；
  **不声明 `zdotc_` 一类复数返回值函数**——Fortran 复数返回 ABI 陷阱，点积用 gemv）。
- `dense_ops.hpp`（**所有权：lead**）：`fill_random`（计数器式 SplitMix64，结果与线程数无关）、
  `dot/norm2/axpy/scal/gram/multiply`、`orthogonalize_cgs2`（欧氏 / 存储 M·V / 按需重算 M·w
  三种模式）。其它模块只消费，不得另写同功能函数。

### 3.4 factor（`include/hsolve/factor.hpp`）
- 唯一后端 MUMPS 5.8（顺序版 + OpenMP）。**一条分解路径且始终有惯性**：实数 SYM=2（LDLᵀ，
  2×2 主元）；复数经实对称嵌入 E（2n 阶，SYM=2）——求解 E[Re x; Im x] = [Re b; Im b]，
  惯性(K) = 惯性(E)/2。（MUMPS 复数 SYM=2 是复对称而非厄密。）
- 接口：`analyze(pencil, α, β)`（以 αA+βB 作代表值，分析期匹配/缩放会用到数值）、`estimate()`、
  `[[nodiscard]] bool factorize(α, β)`、`solve(b, nrhs, ldb)`（多右端一次调用）、`inertia()`、
  `set_out_of_core()`。
- **MUMPS 实测约定**（评审用探针程序在 5.8.2 上验证）：
  - 零主元检测需 `ICNTL(24)=1` **且** `CNTL(3)>0`（`null_pivot_tol`，相对预处理后矩阵范数）；
    CNTL(3)=0 时 3600 阶精确奇异矩阵也会「成功」分解。默认值经 P1 实测标定：双精度 1e-8、单精度
    `single_null_pivot_tol` 1e-4（单精度只能部分检测）；检测是依赖消去顺序的启发式，「近奇异但
    非奇异」的保证相对于 ‖K‖（即 scale）而非 |λ|；漏检时分解成功、位移反演照常工作。
    `deterministic`（ICNTL(48)=0，默认开）保证同线程数下结果可复现。复数时 INFOG(12) 为奇数视为奇异。
  - ICNTL(24)=1 时 MUMPS 不报错而是修补零主元（INFOG(28)），之后的回代会丢掉该方向 ⇒
    **INFO(1)==−10 或 INFOG(28)>0 一律视为奇异：`factorize` 返回 false**，`solve` 拒绝执行，
    `inertia().valid=false`（计数仍填），对象保持已分析状态以便扰动位移后重分解。
  - INFOG(12) 仅在 INFOG(28)==0 时是准确惯性（90000 阶不定 Laplacian 实测与解析值一致）。
  - INFOG 大于 2³¹ 的值以负数、单位 1e6 报告，必须解码；MUMPS 的 MB = 1e6 字节。
- 错误约定：工作区不足（−8,−9,−14,−15,−17,−20）由 `factorize` 内部重试 ≤3 次（ICNTL(14) 翻倍，
  计入 `workspace_retries`）后抛 FactorizationFailed；−13/−19（分配失败 / 超 ICNTL(23) 上限）抛
  OutOfMemoryBudget；其它负 INFO 抛 FactorizationFailed（消息含 INFO(1)/INFO(2)）。
- `scaling` 默认 ICNTL(8)=7（每次分解重算缩放），使一次分析服务不同位移。
- `FactorEstimate`：INFOG(20)（因子条目估计）、INFOG(16/17) 内存、INFOG(26/27) 外存、
  INFOG(36/38) BLR 估计（仅当分析时启用 BLR，否则 −1）、RINFOG(1) flops。
- 单精度 / BLR 分解的 inertia() 是舍入/压缩后矩阵的惯性，**证书不得使用**：driver 在主分解为
  单精度或 BLR 时改用 `compute_inertia`（双精度）计数。
- `compute_inertia(pencil, α, β)`：自建对象、强制双精度且无 BLR、奇异时返回 valid=false 不抛；
  仅用于 B 正定性检测与测试——**driver 的证书复用主分解对象**（同一模式只分析一次）。
  B 为 HPD 当且仅当 `valid && neg==0 && zero==0`。

### 3.4b ops（`include/hsolve/operator.hpp`）
- `CsrOperator`：Y = A X（spmm）。
- `ShiftInvertOperator(factor, pencil, σ, refinement)`：Y = K⁻¹ B X（B = I 时复制）一次多右端回代；
  `refinement.max_steps>0` 时做**双精度迭代修正**：r = BX − (A Y − σ B Y)（两次 spmm，K 不物化），
  Y += K⁻¹ r，直到每列 ‖r‖ ≤ rel_tol·‖BX‖。**单精度或 BLR 分解必须配合修正**（否则特征对只是
  扰动矩阵的特征对，后向误差停在 1e-7 / blr_tol）。单精度仅当 σ 与所有特征值的相对距离
  ≳ 1e-6‖K‖ 时可用（修正收敛需 eps_s·cond(K) < 1），否则退回双精度 out-of-core。
- 构造时记录分解次数，apply 时断言未变（防止同一对象被重分解到别的位移后悄悄改变 Op）。

### 3.5 eigen（`include/hsolve/eigen.hpp`）
- 只依赖 `linear_operator.hpp` 与 `src/linalg`。不因不收敛抛异常（返回 converged=false）。
- 结果契约：m = theta.size()；theta/estimate/vectors 按下标一一对应；[0, nconv) 为已收敛对、
  按 which 排序，[nconv, m) 为未收敛对、同样排序；converged == (nconv ≥ nev)。
- `lanczos_krylov_schur`：算子须在 ⟨x,y⟩_M = yᴴMx 下自伴（M = b_inner 或 I）。CGS2 全正交化
  （默认按需重算 M·w，`store_bv` 才存 M·V），厚重启；**锁定只在耦合达到工作精度
  （|b_l| ≤ ~1e-15‖Op‖）时把 b 分量置零**（按 tol 置零会让其它对的残差估计失真，P1 实测）；
  `deflation` 块 D（M-正交）用于「延迟重启」找更多特征对——此时估计针对收缩算子 P·Op·P，Op x
  在 span(D) 上的分量由 D 自身残差界定，driver 最后对 [D X] 做 Rayleigh–Ritz 消除。m == nev。
  单向量 Krylov 对精确重特征值只能靠舍入得到其余副本：driver 的惯性证书 + 收缩重跑负责补齐。
- `krylov_schur_general`：欧氏内积，实数用实 Schur 形（dgees/dtrsen），共轭对永不拆开：
  m ∈ {nev, nev+1}，共轭对相邻、正虚部在前，nconv 不拆对，schur_form 无被截断的 2×2 块，
  前 nconv 列 Schur 基张成（近似）不变子空间。**不支持 deflation**（非正规算子的不变子空间的
  正交补不是不变的；用法上以更大的 nev/ncv 重跑代替）。
- 两者都通过 `history` 返回每次重启的收敛数与最大残差估计。

### 3.6 driver（`include/hsolve/solver.hpp`）
1. **诊断**：规模、nnz、实/复（A 复 B 实时 `to_complex`）、厄密缺陷（超 `hermitian_tol` 抛错，
   否则 `hermitian_part`）、‖A‖₂/‖B‖₂ 估计（~30 步 Lanczos 的最大 |Ritz 值|：它是 ‖·‖₂ 的**下界**，从而 η 是真实后向误差的**上界**，验收偏保守）、
   scale = ‖A‖₂/‖B‖₂、B 正定性（`compute_inertia(pencil, 0, 1)`）。
2. **策略**：HEP/GHEP → 位移反演 + `lanczos_krylov_schur`（B 内积）；GHIEP → 位移反演 +
   `krylov_schur_general`；谱两端且 k 小 → 不分解，直接 Lanczos。
3. **内存预算**（显式公式）：factor_cap_MB = (budget − current_rss − n·sizeof(T)·(ncv·(1+store_bv)
   + 3k) − margin)/1e6。`analyze` 后估计值 ≤ cap 才分解，并把 cap 作为 ICNTL(23) 传给 MUMPS（超限
   变成 INFO=−19 而不是被系统杀掉）。超预算依次：BLR（需重新分析一次以获得估计）→ 单精度 +
   迭代修正 → out-of-core；仍超则抛 OutOfMemoryBudget 并报告估计值（绝不盲目分解）。
   Interval 模式先用惯性求出区间内个数 m，再定 nev/ncv，n·m·sizeof(T)·3 超限则切片或拒绝。
4. **近奇异位移**：`factorize` 返回 false 时 σ ← σ + δ（δ 从 1e-8·scale 起每次 ×10，最多
   `max_shift_retries` 次），记入报告；「最近 k 个」的选择始终相对原始 σ。
5. **迭代求解**：内层容差 = tol × inner_tol_factor；θ → λ = σ + 1/θ。
6. **精修**：HEP/GHEP 在原问题上做 Rayleigh–Ritz（小规模稠密 hegv），保证 B 正交；GHIEP 对每个
   向量分别计算 θ 映射值与 Rayleigh 商 xᴴAx/xᴴBx 的残差，取残差小者。
7. **残差验证**：每对计算 ‖Ax − λBx‖₂、相对残差、后向误差 η（§4.5）。未达标者做残差修正迭代
   x ← x − K̃⁻¹(A − ρB)x（ρ 为 Rayleigh 商；对任意近似分解 K̃，其不动点都是 (A,B) 的精确特征对）；
   仍不达标则 accepted=false。
8. **惯性证书**（HEP/GHEP；复数经实嵌入同样可用）——窗口按目标类型定义：
   - Nearest：r = max|λᵢ − σ|，a ≤ σ − r、b ≥ σ + r，并各自向外推进到谱隙（借助 ritz_all，
     无信息时用 r(1+δ)）；惯性 valid=false 时移动端点。
   - Smallest：单个计数 ν₋(A − bB) == k，b 取 λ_k 之上的谱隙。Largest：n − ν₋(A − aB) == k。
   - Interval：用户给定的 [lower, upper]。
   - 通过条件 count_inertia == count_found，count_found 为去重后 B-正交的不同向量数；**任何不等
     都重跑**（少 → 以已得向量作 deflation 补齐；多 → 去除重复/幽灵收敛）。
   - 复用主分解对象：factorize(K(a))、factorize(K(b)) 读惯性，仅当需要重跑才重分解 K(σ)。
   - GHIEP 无惯性证书（报告中注明）；改用延迟重启检查：换种子、更大 nev 重跑，确认没有
     |θ| ≥ |θ_k| 的新 Ritz 值收敛。
9. **零模（零空间）处理**：EM 零空间可达 n 的 20% 以上，**不得显式 deflation 或剔除重算**。
   HEP/GHEP：用惯性数出零簇 d = ν₋(A−εB) − ν₋(A+εB)，在两侧各放一个单边位移 σ₊ > ε
   （Which::LargestAlgebraic，θ>0 ⇔ λ>σ₊）与 σ₋ < −ε（SmallestAlgebraic），用惯性二分把 σ± 放在
   使零簇成为单个无关 θ 值的位置，结果按离原 σ 的距离合并。GHIEP 同样用单边代数选择。
   `zero_filter_tol` 定义为相对 scale。
10. **GHIEP 实/复分类**：|Im λ| > complex_tol·max(|λ|, scale) **且** 实投影对未通过后向误差测试而
    复对通过，才算复特征值；复特征对单独输出（`complex_eigenvalues`），实特征对不受影响。
11. **输出**：见 `docs/FORMATS.md §3`。

### 3.7 util（`include/hsolve/log.hpp`、`include/hsolve/util.hpp`）
- `Logger`：JSON lines 写文件 + 人类可读 stderr；事件名小写下划线，字段扁平。
- `Timer`、`peak_rss_bytes()`、`current_rss_bytes()`、`max_threads()`、`build_info()`。

## 4. 数值约定

### 4.1 位移反演
K(σ) = A − σB，OP = K⁻¹B。A x = λ B x ⇔ OP x = θ x，θ = 1/(λ − σ)。离 σ 越近 |θ| 越大。

### 4.2 惯性（Sylvester）
B 正定时，ν₋(A − sB) = 「小于 s 的特征值个数」，由 LDLᵀ 负主元数得到（valid 时）。区间 (a, b)
内特征值数 = ν₋(A − bB) − ν₋(A − aB)。端点须落在谱隙中，否则数值惯性不可靠。

### 4.3 GHIEP
B 不定时惯性计数不成立，理论上可有复特征值。用非厄密 Krylov-Schur；实/复分类见 §3.6-10。

### 4.4 收敛（内层）
Ritz 估计 ≤ tol_inner·|θ|（θ = 0 时用最大 |Ritz 值|）。

### 4.5 残差与精度指标（报告中全部给出）
- 绝对残差 r = ‖A x − λ B x‖₂
- 相对残差 r / (|λ|·‖B x‖₂)（λ = 0 时退化为 r / ‖Bx‖₂）
- **后向误差 η = r / ((‖A‖₂ + |λ|·‖B‖₂)·‖x‖₂)**，‖·‖₂ 取 Lanczos 最大 |Ritz 值|——它是 ‖·‖₂ 的
  **下界**，所以算出的 η ≥ 真实 η（验收偏保守而非偏乐观；1-范数是上界，会使 η 偏乐观，故不用）。
  driver 以 η ≤ tol 为接受标准。scale = ‖A‖₂/‖B‖₂。
- 与参考解比较：特征值相对误差；特征向量用子空间夹角（重特征值按整个簇的子空间比较）。

## 5. 编码规范
- C++17，命名空间 `hs`；文件 `snake_case`；类型 `PascalCase`；函数/变量 `snake_case`；
  私有成员尾随下划线；常量 `kCamelCase`。格式见 `.clang-format`。
- 稠密矩阵列主序；全局索引 `index_t`（int64）；CSR 列索引 `int32_t`。
- 错误：抛 `hs::Error(code, message)`（`HS_THROW`）；输入错误不得用 assert；内部不变量用
  `HS_ASSERT`（所有构建类型都启用）。
- 每个公共函数在头文件注明：前置条件、复杂度、线程安全性、是否修改输入；每个头文件顶部注明
  标量集合。
- **显式实例化**：模板只在其所属模块的 .cpp 底部用 `HS_FOR_EACH_SCALAR` 实例化一次。
- **符号归属**：声明在 `include/hsolve/<m>.hpp` 的符号只能定义在 `src/<m>/` 下（显式实例化是弱
  符号，重复定义会被链接器静默挑一个）。临时替身只能写在测试文件的匿名命名空间里且换名。
- 非拥有引用的构造函数删除右值重载（防悬垂）；选项结构里的指针只在调用期间有效。
- **禁止**：稀疏矩阵稠密化（唯一例外：测试/评测代码中 n ≤ 4000 的参考解）；全局可变状态；
  无种子随机数；库代码中 `std::cout`/`printf`。
- 测试：`tests/cpp/test_<module>*.cpp`（GoogleTest），**套件名以模块名开头**（`Sparse*`、`Io*`、
  `Factor*`、`Ops*`、`Lanczos*`、`KrylovSchurGeneral*`、`Driver*`）；ctest 名为
  `<可执行文件>.<套件>.<用例>`，`tools/dev/test.sh release test_factor` 只跑该文件，匹配不到测试即
  报错。共享夹具在 `tests/cpp/support/`（`hsolve_test_support`），只追加不改签名。
  数值测试尽量用有解析解的问题。Python 工具测试在 `tests/py/`（pytest）。
- 每个模块的测试须调用其头文件声明的每个模板的每个标量实例（缺实例化在模块内就暴露）。
- 新增依赖必须是开源许可，并登记到 `docs/DEPENDENCIES.md`。

## 6. 资源与实验规范
本机为共享服务器（2×Xeon Gold 6336Y，48 物理核，125 GB）。**所有编译之外的运行都必须
经过 `tools/run/guard.py`**：
- 默认绑定 NUMA node1 的 16 个物理核 `24-30,33-41`，进程树 RSS 上限 30 GiB；系统可用内存低于
  12 GiB 时只终止「大」进程树（≥ max(2 GiB, mem_gb/4)），小的构建/测试不受连带。
- 峰值内存报告取采样和、内核高水位 VmHWM 之和、ru_maxrss 三者最大值（不漏短时尖峰）。
- 重型实验加 `--exclusive`（全机串行）；单元测试用 `--skip-busy-check`。
- 多 agent 并行阶段：每个 agent 使用不相交的 `HS_CORES`（各 2 核）与 `HS_JOBS=2`，并设置
  `HS_MAX_SHARED_MEM_GB=8`（超过 8 GiB 的运行必须 `--exclusive`）。
- 预估内存（MUMPS 分析）超过 28 GiB 的实验不运行，记录并上报。

## 7. 版本管理
- `main` 始终可编译、测试全过。功能在分支（或 worktree）开发，合并前须通过评审与测试。
- 提交信息：`<module>: <summary>`（如 `factor: add single-precision MUMPS backend`）。
- 生成数据（`data/`）、构建目录与原始实验输出不入库；实验汇总（`results/*.md|json`）入库。
- 文件所有权（P1）：`include/hsolve/<m>.hpp` 与 `src/<m>/` 归该模块；`src/linalg/*`、
  `tests/cpp/support/*`、`tools/case.py`、`CMakeLists.txt`、`docs/*` 归 lead——其他人需要改动时
  在交付说明中提出。

## 8. 路线图
| 阶段 | 内容 |
|---|---|
| P0 | 仓库骨架、环境、资源看门狗、接口与设计文档；设计评审 v2 |
| P1 | io / factor+ops / eigen（两种）实现与单测；测例生成器；评测工具 |
| P2 | driver + CLI；端到端测试；测例集与首轮实验（对比 scipy 基线） |
| P3 | 性能剖析与优化（块回代、SpMV AVX-512、MUMPS 参数、混合精度、正交化 DGKS/BLAS3）；LOBPCG / JD；区间切片；零簇移位放置；GHIEP 重特征值（块 Krylov） |
| P4 | 故障场景验证矩阵、方案文档、复现材料 |
