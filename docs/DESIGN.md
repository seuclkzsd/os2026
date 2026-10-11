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

v3（P3 合并后，a8ed8f4）——低内存输入装配、无散射映射的铅笔；确定性正交化核（DGKS、融合扫描、
块 CGS2）；MUMPS 实测约定补充（实例间非线程安全、ICNTL(31)、PERM_IN、JOB=−4）与 InertiaCounter；
块 Krylov-Schur、LOBPCG、Jacobi-Davidson；driver 的谱切片、惯性搜索的单边目标、零带切除、GHIEP 零簇
移位放置、互斥分阶段计时；Python 接口。实测数据汇总在 `results/p3_measurements.md`。

v4（P4 合并后，6a3d7c1）——稀疏核运行时分派（Reference / Portable / AVX-512）与融合的修正残差
`shifted_residual`；修正只解活跃列、到舍入水平即停；`FactorSolveOperator`；自研并行嵌套剖分排序
（factor 层，driver 尚未使用）；一般 Krylov-Schur 的块模式与 `krylov_schur_general_peak_columns`；
新模块 src/verify（随机排除检验、单精度计数规则、代价模型；尚未接入 driver）；driver 的块 Krylov
（`block_size`）、InertiaCounter 证书计数与第五个互斥阶段 certificate、确定性的排序代价模型、B 的
O(nnz) 正定性筛查、GHIEP 块运行与后处理内存；集群节点部署。实测汇总在 `results/p4_measurements.md`。

---

## 1. 问题范围

| 类型 | 形式 | 说明 |
|---|---|---|
| HEP | A x = λ x | A 厄密（实对称或复厄密），通常不定 |
| GHEP | A x = λ B x | A 厄密，B 厄密正定（质量矩阵） |
| GHIEP | A x = λ B x | A、B 厄密，B 不定。宣讲真实算例即此类（Lee–Sun–Cendes 波导公式，λ = −β²） |

- 规模：目标 n ≈ 1e5，测试覆盖到 1e6；nnz 可达 1e8 量级。
- 求解目标：σ 附近最近的 k 个；区间 [a, b] 内全部；代数最小 / 最大的 k 个（后三者需要惯性计数，
  B 不定时只支持最近 k 个）。
- 硬约束（赛题红线）：**禁止把稀疏矩阵显式稠密化**；禁止硬编码答案、读取参考结果、针对
  公开用例调参。所有算法参数必须由矩阵性质与用户配置按文档化规则推导。
- 运行环境：单机多核 CPU（≥8 核），内存 16 / 32 GB；GPU 可选。

## 2. 总体架构（严格分层，依赖只能自上而下）

```
apps/hsolve_cli            命令行：读配置 → 调 driver → 写结果（docs/FORMATS.md §3）
python/                    pybind11 进程内接口 hsolve._hsolve + hsolve 包（docs/PYTHON.md）
        │
src/driver   诊断、策略、位移管理、谱切片与单边目标、零簇处理、惯性证书、精修、残差验证、报告(JSON)
        │
src/verify   完备性检验构件：随机排除检验、单精度计数规则、代价模型（verify.hpp；依赖 eigen、factor、
        │    sparse、linalg；driver 尚未调用，P5 接入）
src/eigen    厚重启 Lanczos（厄密 Krylov-Schur）与一般 Krylov-Schur（GHIEP），二者都有块模式
        │    block_size；LOBPCG、Jacobi-Davidson（eigen_iterative.hpp）
        │        ↑ 只依赖 include/hsolve/linear_operator.hpp、eigen.hpp 与 src/linalg
src/ops      LinearOperator 实现：CsrOperator（A）、ShiftInvertOperator（(A−σB)⁻¹B，可选迭代修正）、
        │    FactorSolveOperator（K⁻¹，预条件子用）
src/factor   MUMPS：分析/估计/分解/多右端回代/惯性；复数走实嵌入；InertiaCounter（只计惯性）；
        │    自研并行嵌套剖分 compute_ordering（以 PERM_IN 交给 MUMPS）
src/sparse   CsrMatrix 工具、SpMV/SpMM 与融合残差 shifted_residual（运行时分派：Reference / Portable /
             AVX-512）、ShiftedPencil（上三角并模式，无散射映射）、实嵌入、厄密化
src/linalg   DenseMatrix、BLAS/LAPACK 封装、随机向量、确定性归约、CGS2 / 块 CGS2 正交化（dense_ops）
src/io       npy / npz / mtx 读写（流式、低内存装配）、三角存储恢复、厄密性检查与厄密化
src/util     错误、日志(JSON lines)、计时、内存(RSS)
```

- 下层模块（io / sparse / linalg / factor / ops / eigen / verify）**不打印、不依赖日志**；统计信息
  通过返回的结构体（`FactorStats`、`InertiaCounterStats`、`*EigResult::history`、`ExclusionResult`）
  交给 driver 统一记录。
- 头文件：`include/hsolve/` 为公共接口；`src/<module>/*.hpp` 为模块内部头文件。
- `python/` 只用公共接口 `hs::solve` / `hs::load_matrix`：与 CLI 同一条求解路径、同一份配置格式；
  scipy/numpy 输入一次转换为 `CsrMatrix`（不稠密化），特征向量零拷贝交给 numpy，求解期间释放 GIL。
  用法、拷贝策略与实测开销见 `docs/PYTHON.md`。

## 3. 模块规格

### 3.1 io（`include/hsolve/io.hpp`）
- `.npy`：版本 1.0/2.0/3.0；`parse_npy` 接受任意定长标量 descr（`<f8 <c16 <i4 <i8 |b1 |u1 <u8
  |S3 <U3`…，含大端），保留原始字节；类型转换只在 `npy_to_vector` 中进行（拒绝不支持的转换）。
  结构化/对象 dtype 拒绝。
- `.npz`：zip（stored / deflate，zlib），**成员大小与偏移取自中央目录及其 zip64 扩展字段**
  （numpy 以 force_zip64 写成员，本地头里是 0xFFFFFFFF），校验 CRC-32。成员直接解压进目标数组
  （无整档缓冲）；需要类型转换的成员（f4/c8 值、int64 索引）与 coo/csc 的成员按 ~1 MiB 分块流式
  读入（`NpzFile::read_chunks`），转换错误在该成员的 CRC 校验之后报告。头部声明的元素数先与成员
  实际长度核对再分配（声称 2⁴⁰ 个元素的小文件报 Io，不会先去分配）；deflate 成员声明的解压尺寸
  不得超过压缩字节数的 1032 倍（deflate 的最大膨胀率）。
- scipy `save_npz`：csr / csc / coo，matrix 与 array 两种 API；忽略格式不用的键（如 `_is_array`
  `|b1`）；`format` 键可能是 `|S3` 或 `<U3`；索引 int32/int64；bsr / dia 抛 Unsupported。
- `.mtx`：coordinate（real/complex/integer/pattern × general/symmetric/hermitian；skew 报错）与
  array 格式。coordinate 文件**读两遍**：先数每行条目，再把每个条目直接放进它的 CSR 槽位（两遍
  之间以行索引指纹核对文件未变）；错误仍按文件顺序报告。
- 稠密 `.npy` 二维数组允许作为输入（小矩阵），转换为 CSR 时丢弃精确零。
- **低内存装配**（`src/io/csr_assembly.hpp`）：不经 int64 三元组。csr 读入后原地规范化（行内稳定
  排序、按文件顺序合并重复项、去精确零；变短的行把新长度记在空出的槽位，无行长数组）；coo / csc
  只保留行索引，由 `io::RowSlots`（计数排序，行指针兼作游标，每行 8 B）把列索引与随后流入的值直接
  放进 CSR 槽位，无三元组副本、无转置。峰值 / 结果（实 / 复；`bench_memio_load` 每行 26 项、
  `bench_memio_fix_rows` 每行 1–3 项，字节公式见 `src/io/load.cpp`）：npz csr 1.0×；npz csc/coo
  1.34× / 1.20×（对角矩阵 1.22×）；MatrixMarket coordinate 1.0× + 4 MiB 行缓冲；单三角存储 1.22× /
  1.34×（三对角 1.55× / 1.59×）；结构不对称输入取厄密部分 1.70× / 1.82×。P2 时为：csc/coo 全存储
  2.02–2.35×，单三角 2.5×，厄密部分 3.0×。行越稀疏，coo/csc 与单三角的比值越高。
- 加载流水线：建 CSR（合并重复项）→ 拒绝非有限值 → 三角存储恢复（`Storage::Auto`：严格下三角空且
  严格上三角非空视为上三角存储，反之亦然；用户声明 Upper 却出现下三角元素则报错；
  `symmetrize_from_triangle(&&)`）→ 厄密检查（缺陷 ‖A−Aᴴ‖_F/‖A‖_F 用缩放平方和计算，任意量级都
  准确；超 `hermitian_tol` 抛 NotHermitian）→ **按值比较不是精确厄密（某个 a_ij ≠ conj(a_ji)）时以
  `hermitian_part(&&)` 替换**（结构对称时原地）并记入 notes（保证分解只读一个三角、算子与残差用
  全矩阵时看到的是同一个矩阵）→ 全零虚部的复数输入降为实数。
- 输出：`write_npy_vector`、`write_npy_matrix`（Fortran 序，np.load 得 (rows, cols)）。

### 3.2 sparse（`include/hsolve/sparse.hpp`）
- `CsrMatrix<T>` 不变量见头文件；列索引 int32（n ≤ 2³¹），行指针 int64。
- `csr_from_coo / conj_transpose / add / identity / triangle / symmetrize_from_triangle /
  hermitian_part / to_complex / norm_fro / norm_one / hermitian_defect / spmv / spmm /
  real_embedding`。
- `symmetrize_from_triangle`、`hermitian_part` 单遍构造（先计数，结果一次分配）；它们与
  `to_complex` 另有消费输入的右值重载，输入数组用完即释放：单三角恢复峰值约为结果的 1.2×（实）/
  1.3×（复）；厄密部分在结构对称时原地、零分配，否则 1.7× / 1.8×（P2 约 2.5× 与 3×）。
  `hermitian_part` 与 `add(0.5, A, 0.5, Aᴴ)` 逐位一致；`symmetrize_from_triangle` 原样存 u_ij 与
  conj(u_ij)（旧写法乘 (1+0i) 会改动复数镜像的带符号零，二者按值相等）。
- `ShiftedPencil<T>`：A ∪ B 的**上三角并模式（只存索引，对角线必在每行首位）**，外加每行 1 字节
  标志（A 或 B 的上三角行是否恰为整行模式）。`assemble(α, β, U* vals)` 逐行即时归并 A、B 的有序
  上三角行（整行者直接复制），直接把 αA + βB 写入后端的值数组（U 可为单精度），全量 K 从不物化；
  **无散射映射**，assemble 不分配内存，算术同旧实现（值逐位一致）。内存 8(n+1) + 4·nnz + n 字节。
  实测（`bench_memio_pencil`，旧 → 新）：wg-square-1e6 保留内存 112.5 → 33.2 MB、assemble 42.0 →
  24.3 ms、构造 0.137 → 0.063 s；ss-tem152078 69.8 → 14.6 MB；B 远比 A 稀疏时走归并路径，assemble
  4.1 → 5.7 ms（约一次分解的 0.1%）。
  K(σ) = 1·A + (−σ)·B，B 自身 = 0·A + 1·B，A 自身 = 1·A + 0·B，同一模式服务所有位移、B 正定性检测
  与 A 的零主元计数。`assemble_full` 仅供测试/小问题。
- **稀疏核与运行时分派**（`src/sparse/kernels.cpp`，P4）：spmv / spmm / `shifted_residual` 有
  `KernelPath`：Reference（P1 循环：跨步 spmm，每遍 8 个向量）、Portable（交错块内核，按块宽 1..16
  分别编译，累加器为构建 ISA 宽度的 GCC/Clang 向量扩展）、Avx512（AVX-512F intrinsics，运行时以
  `__builtin_cpu_supports` 检测，不可用时显式请求抛 Unsupported）。Auto = 每行 ≥ 8 个 double 的块
  （实数 q ≥ 8、复数 q ≥ 4）用 AVX-512，否则 Portable（窄块上 Portable 更快，`results/p4_measurements.md`
  §1）。环境变量 `HS_KERNELS_NO_AVX512`（首次调用时读一次）屏蔽 AVX-512，用于在 AVX-512 机器上测试回退。
  - 多向量（p ≥ 2）先把 X 拷成行主序交错块（每遍 A 服务 16 个向量，连续读），工作区 n·min(p, 16)
    标量，可由 `KernelOptions::workspace` 复用；A 每列平均非零 < 2（p ≤ 4）或 < 8（更宽）时 spmm 保留
    Reference 跨步循环（如对角 B）；单个向量在所有路径上都走 Reference 行循环。
  - **确定性**：每个输出行由一个线程按固定顺序计算 ⇒ 同一路径的结果与线程数无关、逐次运行逐位相同；
    不同路径舍入不同（求和顺序、FMA），对 Reference 的差 ≤ 1e-14·（|A||X| 尺度）（测试契约，典型
    ~1e-15），不逐位一致。
  - `shifted_residual(A, B, σ, X, Y, R, p, …, ω)`：R = BX − (AY − σBY)（B = nullptr 即 I），快路径按
    B(X + σY) − AY 一次扫描 A、B（Reference 为三次 spmm）；可选输出每列分量后向误差
    ω_k = maxᵢ |r_ik| / (|B||w| + |A||y|)ᵢ（w = x + σy；复数 |z| = |Re z| + |Im z|；0/0 = 0；r/0 = inf；
    任一行 r 或分母非有限 ⇒ ω_k = inf），同一次扫描算出，max 精确故与线程数无关。ω 为几个 eps 时
    残差就是自身求值的舍入噪声。

### 3.3 linalg（`include/hsolve/dense.hpp`、`src/linalg/`）
- `DenseMatrix<T>`：列主序、64 字节对齐、`ld == rows`。
- `blas_lapack.hpp`：自行声明 Fortran BLAS/LAPACK（LP64，隐藏字符串长度参数显式传递；
  **不声明 `zdotc_` 一类复数返回值函数**——Fortran 复数返回 ABI 陷阱，点积用 gemv）。
- `dense_ops.hpp`（**所有权：lead**）：`fill_random`（计数器式 SplitMix64，结果与线程数无关）、
  `dot/norm2/axpy/scal/gram/multiply`、`orthogonalize_cgs2`（欧氏 / 存储 M·V / 按需重算 M·w
  三种模式）、`orthogonalize_block_cgs2`。其它模块只消费，不得另写同功能函数。
- **确定性归约**：`dot`、`norm2` 与 CGS 的投影 Vᴴ M w 都在只依赖问题规模的固定行块上求和（块内
  显式通道累加器、不重结合；块和按块序累加进至多 256 个槽，槽再串行相加）⇒ 任意 OpenMP 线程数、
  任意次运行**逐位相同**（同一二进制；换编译器、FMA 收缩等编译选项或 BLAS 可改变末位）。CGS 核
  因此手写（GCC/Clang 向量扩展），不用 BLAS gemv。
- `orthogonalize_cgs2(…, const OrthoOptions&, OrthoInfo*)`：
  - `Reorth`：`Always`（两遍，经典 CGS2）、`IfNeeded`（DGKS：第一遍后 ‖w‖_M ≥ η·‖w_前‖_M 则跳过
    第二遍；η 默认 1/√2，同 ARPACK/SLEPc）、`Auto`（**默认**：欧氏与重算模式用 DGKS——范数由例程
    自己算；存储模式两遍——范数要取自调用者的 bw，而 bw 可能只是草稿）。
  - **DGKS 的前提**：V 在工作精度上 M-正交归一。若 ‖VᴴMV − I‖ = δ，一遍后 VᴴMw 仍为 O(δ)，两遍为
    O(δ²)。因此带 deflation 块的调用方用 `Always`：`lanczos_krylov_schur` 在 d > 0 时（deflation 块
    只在 1e-8 内正交归一）、块正交化器的逐列回退路径（6ff42f2）。
  - `OrthoOptions{reorth, eta, mw_on_exit}`：`mw_on_exit = false`（仅重算模式）声明调用者出口不需要
    bw == M·w，范数改用勾股关系，M 作用一遍 2 → 1 次、两遍 3 → 2 次；默认 true，尚无调用者使用
    （数据算例的 B 都便宜，测不出收益）。`OrthoInfo{passes, norm_before, norm_pass1, norm_after}`。
  - 融合扫描：第一遍的更新与第二遍的投影在同一行块驻留 L2 时完成。两遍：欧氏 3 次扫描 V，存储
    1 次 BV + 2 次 [V BV]，重算 4 次 V + 3 次 M 作用；一遍：2、1 + 1、2（+ 2 次 M）。原 BLAS2 实现
    为 4、6、4。并行扫描每次至少 64 个行块（负载均衡）。
- `orthogonalize_block_cgs2(…, ApplyBlockFn)`：W（p 列）对 V 的块 CGS2（BLAS3 gemm，在 OpenMP 区域
  内按固定行块调用、块内单线程），三种模式同上，DGKS 按块判定（任一列不满足即整块做第二遍）；W 的
  列之间**不**正交化（调用者再做块内 QR）。只要单线程 gemm 确定，结果与线程数无关。工作区上界
  `orthogonalize_block_cgs2_workspace_bytes(k, p)`（≈ max(2 MB, 32·8(k·p·d + p))，d = 1 实 / 2 复）。
  n = 2e5、k = 64、p = 8、两遍时比 8 次逐列调用快 2.9–4.6×（对旧 BLAS2 实现 3.5–7.0×）。
- eigen 的 `BlockOrthonormalizer`（`src/eigen/block_ortho.*`：块 CGS2 + M 内积 Cholesky-QR，两遍，
  块 Krylov-Schur 专用；一般 Krylov-Schur 块模式的回退核）与 `eigen_detail::block_cgs_cholqr2`
  （`src/eigen/block_general.*`：本节的 `orthogonalize_block_cgs2` + 两次 Cholesky-QR，一般块模式的
  快路径，§3.5）是同类核，暂归 eigen；是否并入 dense_ops 由 lead 决定。

### 3.4 factor（`include/hsolve/factor.hpp`）
- 唯一后端 MUMPS 5.8（顺序版 + OpenMP）。**一条分解路径且始终有惯性**：实数 SYM=2（LDLᵀ，
  2×2 主元）；复数经实对称嵌入 E（2n 阶，SYM=2）——求解 E[Re x; Im x] = [Re b; Im b]，
  惯性(K) = 惯性(E)/2。（MUMPS 复数 SYM=2 是复对称而非厄密。）
- 接口：`analyze(pencil, α, β)`（以 αA+βB 作代表值，分析期匹配/缩放会用到数值）、`estimate()`、
  `[[nodiscard]] bool factorize(α, β)`、`solve(b, nrhs, ldb)`（多右端一次调用）、`inertia()`、
  `set_out_of_core()`、`set_null_pivot_detection(on)`、`release_factors()`。
- **MUMPS 实测约定**（评审用探针程序在 5.8.2 上验证）：
  - 零主元检测需 `ICNTL(24)=1` **且** `CNTL(3)>0`（`null_pivot_tol`，相对预处理后矩阵范数）；
    CNTL(3)=0 时 3600 阶精确奇异矩阵也会「成功」分解。默认值经 P1 实测标定：双精度 1e-8、单精度
    `single_null_pivot_tol` 1e-4（单精度只能部分检测）；检测是依赖消去顺序的启发式，「近奇异但
    非奇异」的保证相对于 ‖K‖（即 scale）而非 |λ|；漏检时分解成功、位移反演照常工作。
    复数时 INFOG(12) 为奇数视为奇异。
  - `FactorOptions::deterministic`（ICNTL(48)=0，factor 层默认 true）使同一线程数下的分解与回代
    逐位可复现；**driver 默认 `SolverConfig::deterministic = false`**（ICNTL(48)=1，L0-OpenMP 子树
    并行：三角回代约快 1.75×，结果只在容差内可复现，574d114）。L0-OpenMP 下分解必须用其分析时的
    线程数，否则 INFO(1) = −58：`analyze()` 固定 ICNTL(16)（threads = 0 取分析时的
    `omp_get_max_threads()`）。逐位复现还需固定排序（下条与 §3.6-3b）。
  - ICNTL(24)=1 时 MUMPS 不报错而是修补零主元（INFOG(28)），之后的回代会丢掉该方向 ⇒
    **INFO(1)==−10 或 INFOG(28)>0 一律视为奇异：`factorize` 返回 false**，`solve` 拒绝执行，
    `inertia().valid=false`（计数仍填），对象保持已分析状态以便扰动位移后重分解。
  - INFOG(12) 仅在 INFOG(28)==0 时是准确惯性（90000 阶不定 Laplacian 实测与解析值一致）。
  - INFOG 大于 2³¹ 的值以负数、单位 1e6 报告，必须解码；MUMPS 的 MB = 1e6 字节。
- **P3 补充的 MUMPS 实测约定**（certpar / memio，`bench_certpar_inertia`、`bench_certpar_fix_probes`）：
  - **顺序版 MUMPS 实例之间不是线程安全的**：分解共用进程级 Fortran 模块状态（MUMPS_LOAD_INIT
    数组、通信缓冲），两个实例在两个线程上并发分解时 10/10 次崩溃（MUMPS_LOAD_INIT，SIGSEGV）。
    因此本进程**所有 MUMPS 调用（JOB −1/−2/−4/1/2/3）由一把全局锁串行**（`hs::mumps::call_mutex()`，
    直接用 pthread mutex：在不链接 libpthread 的 ASan 构建里 std::mutex 是空操作）。不同对象可在
    不同线程上使用，但得不到并行收益。这是本项目唯一的进程级可变状态。
  - ICNTL(31)=1（丢弃因子）在**分析时**读取，只在分解前设置无效；开启后只留主元符号（INFOG(12)、
    INFOG(28)），工作内存 INFOG(21) 降 41–61%、分解时间降 10–23%（实数 1e5 测例，2 线程；低填充的
    复数实嵌入无收益），JOB=3 回代报 INFO(1) = −44。
  - JOB=−4 释放因子、保留分析，之后 JOB=2/3 照常可用。
  - 给新实例以前一次分析的主元顺序（SYM_PERM → PERM_IN，ICNTL(7)=1）：SCOTCH（n ≈ 1e5）分析从
    0.9–1.3 s 降到 0.03–0.05 s；对 AMD 几乎无益（读入输入占主导）。消去不完全复现（MUMPS 重建消去树）：
    1e5 测例因子条目 +0.4..3.4%，随机稀疏矩阵 +1..7%（AMD、SCOTCH）、最多 +15%（METIS）。
  - **SCOTCH 排序不可复现**：同一进程内两次分析得到不同排序（flops 差 10%；factor.hpp 称新进程
    能复现）；memio 在不同进程间也观察到差异（wg-square-1e6 固定 ordering=3，4 次运行因子条目
    7.202e7–7.275e7；ss-ge87h76 两个进程的估计 5.80e8 与 5.91e8）。`deterministic` 只覆盖分解与回代，
    不覆盖排序。
- 错误约定：工作区不足（−8,−9,−14,−15,−17,−20）由 `factorize` 内部重试 ≤3 次（ICNTL(14) 翻倍，
  计入 `workspace_retries`）后抛 FactorizationFailed；−13/−19（分配失败 / 超 ICNTL(23) 上限）抛
  OutOfMemoryBudget；其它负 INFO 抛 FactorizationFailed（消息含 INFO(1)/INFO(2)）。
- `scaling` 默认 ICNTL(8)=7（每次分解重算缩放），使一次分析服务不同位移。
- `FactorEstimate`：INFOG(20)（因子条目估计）、INFOG(16/17) 内存、INFOG(26/27) 外存、
  INFOG(36/38) BLR 估计（仅当分析时启用 BLR，否则 −1）、RINFOG(1) flops。
- 单精度 / BLR 分解的 inertia() 是舍入/压缩后矩阵的惯性，**证书不得使用**：driver 在主分解为
  单精度或 BLR 时改用 `compute_inertia`（双精度）计数。
- `compute_inertia(pencil, α, β)`：自建对象、强制双精度且无 BLR、丢弃因子（ICNTL(31)，只占工作
  内存）、奇异时返回 valid=false 不抛；用于主分解为单精度/BLR 时的计数、B 正定性检测与测试。
  B 为 HPD 当且仅当 `valid && neg==0 && zero==0`。
- `release_factors()`（非纯虚，默认返回 false；MUMPS：JOB=−4）：释放数值因子与工作区、保留分析。
  之后 `solve` 抛 Internal、`inertia().valid = false`（valid ⇔ solve 可用，ShiftInvertOperator 与
  driver 都以它判断），计数与 `stats()` 仍描述被释放的那次分解——需要时先拷贝 `inertia()`。
- `InertiaCounter<T>(analysed_source, options)`：在已分析主对象的模式上计 ν(αA + βB)，**不动主对象**
  （主因子保持有效，证书后不必在工作位移重分解）。机制：自有的只计惯性 MUMPS 实例（ICNTL(31)=1），
  用主对象的主元顺序分析（PERM_IN，不再做一次填充排序）；`count(α, β, detection, precision)`、
  `count_pair(…)`（两次依次计数——并发不可行，见上；diag(K(b), −K(a)) 一次分解慢 1–10%、内存 1.5×）、
  `estimate()`、`stats()`。实测（2 线程，主因子常驻，driver 路径 = 在两端点重分解主对象，5 次交错
  中位数）：3-D SCOTCH 测例证书对 −12..−15%（syn-mem27-46 3.85 → 0.06 + 3.27 s），2-D AMD 测例 −2%，
  低填充的复数实嵌入（syn-near-singular-complex-1e5）+42%（计数器的分析 0.48 s 与 AMD 分析相当）。
  一次计数的工作内存 INFOG(21) 为主因子的 39–59%（例 289 vs 683 MB）；主因子常驻时峰值 RSS 增加
  9–228 MB，先 `release_factors()` 则不增加。
  精度：双精度计数对距位移远于 O(eps·‖K‖·增长因子) 的特征值精确；零主元检测通常标出更近的位移，
  但它是同一个依赖消去顺序的启发式，可能漏检（计数器按自己的顺序消去，漏在与主对象不同的位移上：
  2-D Laplacian 24 个精确特征值位移中漏 1 个，漏检的计数落在 [#below, #below + 重数] 内），故
  valid == true 只在位移位于谱隙时才是证书——端点仍须放在谱隙中点。单精度计数（smumps；3-D 测例快
  45–53%）是被 O(eps_s·‖K‖) 扰动矩阵的精确计数：距最近特征值 δ·(scale + |s|) 内会数错（实测到
  δ ≈ 1.6e-6，稠密复谱 5e-6；δ ≥ 1e-5 从未出错），只能作筛查值。
- **driver 的证书目前仍复用主分解对象**（§3.6-8：端点处重分解主对象读惯性，需要时再回到 σ 重分解；
  主分解为单精度/BLR 时用 `compute_inertia`）。InertiaCounter 的接入在 P4 进行中，计划规则见 §3.6-8。

### 3.4b ops（`include/hsolve/operator.hpp`）
- 稀疏乘积走 §3.2 的内核，构造参数 `KernelPath kernels`（默认 Auto；Reference 只换稀疏乘积，修正
  策略在所有路径上都是 P4 的，因此开修正时没有路径复现 P1 算子——`bench/solvephase_fix_refine.cpp`
  有其副本）；各算子在两次 apply 之间保留内核工作区。
- `CsrOperator`：Y = A X（spmm）。
- `ShiftInvertOperator(factor, pencil, σ, refinement, kernels)`：Y = K⁻¹ B X（B = I 时复制）一次多
  右端回代（回代占一次 apply 的 80–100%，p = 16 时每列代价是 p = 1 的 1/4–1/9，`results/p4_measurements.md`
  §1）；`refinement.max_steps > 0` 时做**双精度迭代修正**：r = B(X + σY) − A Y（shifted_residual：
  一次融合扫描 A、B，K 不物化，同时给出每列分量后向误差 ω；非有限残差 ω = inf），修正只对仍活跃的
  列做块求解 Y += K⁻¹ r；列在以下任一条件时停止：‖r‖ ≤ rel_tol·‖BX‖；ω ≤ 8 eps
  （`SolveRefinement::stop_at_rounding_level`，默认开；false 恢复 P1 的停止规则，修正仍只解活跃列）；
  或修正未降低 ‖r‖（丢弃该修正，保留最优迭代）。每次 apply 至多 max_steps 次块修正。rel_tol 不可达时
  返回列可能 ‖r‖ > rel_tol·‖BX‖，调用方须读 `last_relative_residual()` / `last_backward_error()`。
  **单精度或 BLR 分解必须配合修正**（否则特征对只是扰动矩阵的特征对，后向误差停在 1e-7 / blr_tol）。
  单精度仅当 σ 与所有特征值的相对距离 ≳ 1e-6‖K‖ 时可用（修正收敛需 eps_s·cond(K) < 1），否则修正
  停滞或发散（实测 syn-mem27-46、σ = 0：10 步后 ‖r‖/‖Bx‖ ≈ 5e-6）。**driver 目前不检查**
  `last_relative_residual()`，也不会自动换精度（§3.6-3、§3.6-5）。修正工作区 2·n·p 标量。
- `FactorSolveOperator(factor)`（P4）：y = K⁻¹x，一次 p 列回代（任意精度的因子，不乘 B、不修正），
  如 LOBPCG / JD 的预条件子；构造检查与重分解防护同 ShiftInvertOperator。测试与 bench 中的局部副本
  （`test_jacobi_davidson.cpp`、`bench/lobpcg_jd_memory.cpp`）尚未换成它。
- 构造时要求可用的分解（`inertia().valid`，否则 InvalidInput），记录分解次数，apply 时断言未变
  （防止同一对象被重分解到别的位移后悄悄改变 Op，违反抛 Internal）。

### 3.5 eigen（`include/hsolve/eigen.hpp`、`include/hsolve/eigen_iterative.hpp`）
- 只依赖 `linear_operator.hpp`、`eigen.hpp` 与 `src/linalg`。不因不收敛抛异常（返回 converged=false）。
- 结果契约（四个求解器共用）：m = theta.size()；theta/estimate/vectors 按下标一一对应；[0, nconv)
  为已收敛对、按 which 排序，[nconv, m) 为未收敛对、同样排序；converged == (nconv ≥ nev)。
- `lanczos_krylov_schur`：算子须在 ⟨x,y⟩_M = yᴴMx 下自伴（M = b_inner 或 I）。CGS2 全正交化
  （默认按需重算 M·w，`store_bv` 才存 M·V），厚重启；**锁定只在耦合达到工作精度
  （|b_l| ≤ ~1e-15‖Op‖）时把 b 分量置零**（按 tol 置零会让其它对的残差估计失真，P1 实测）；
  `deflation` 块 D（M-正交）用于「延迟重启」找更多特征对——此时估计针对收缩算子 P·Op·P，Op x
  在 span(D) 上的分量由 D 自身残差界定，driver 最后对 [D X] 做 Rayleigh–Ritz 消除。m == nev。
  正交化用 `Reorth::Auto`（重算模式下为 DGKS：每步 2 或 3 次 M 作用）；有 deflation 块时一律两遍
  （`Always`，每步 3 次；§3.3 的 DGKS 前提）。单向量 Krylov 对精确重特征值只能靠舍入得到其余
  副本：driver 的惯性证书 + 收缩重跑负责补齐。block_size = 1 与 P2 的求解器逐位一致
  （`test_lanczos_bitwise.cpp` 对 base 17a2a79 钉住每种标量 12 种配置的输出哈希）。
- **块模式**（`KrylovSchurOptions::block_size = p > 1`，块 Krylov-Schur，Zhou & Saad 2008）：每步
  一次 p 列的 `op.apply`（多右端回代）；新块对 [D V] 做块 CGS2、块内做 M 内积 Cholesky-QR
  （`BlockOrthonormalizer`：G 非正定、R 的列尺度条件数 > 1e5 或某列在两遍后仍相关时回退逐列 CGS2，
  相关列换成确定性随机向量——块版的 Lanczos 中断）；投影矩阵为块三对角加重启的箭形块，估计为
  ‖R s_last‖₂。起始块：`initial` 的前 min(p, cols) 列 + 随机列。
  - ncv：基数 = 单向量的 ncv（自动或给定）上取整到 p 的倍数，且 ≥ s + 2p（s = nev + 1 上取整到 p 的
    倍数）；给定 ncv 时即为基数，自动时再加 4p（受 n − d ≥ 2(ncv + p) 限制）。每次重启保留
    k = nev + (ncv − nev)/2 下取整到 p 的倍数，至少 s（留一个备用 Ritz 对）、至多 ncv − p。
  - **提前停止探测**：在块边界也做 Rayleigh–Ritz（第一轮从基数处起，之后从 k + p 起），nev 个全部
    收敛即结束本轮。块运行因此恰好达到 tol，而 p = 1 只在轮末检查、通常超出 tol 几个量级——两者
    须在相同残差下比较，不能在相同 tol 下比较。
  - 自由空间不足（n − d < 2(ncv + p)），或 p 大到放不下一个块基，则用一轮覆盖整个自由空间的单向量
    循环精确求解（n − d 次作用）。
  - 重数 ≤ p 的特征值的全部副本一轮内找到；超出 p 的副本只能靠舍入进入（同 p = 1）。
  - 内存：基 n·(d + ncv + p) 标量（store_bv 时加倍），无 store_bv 时另加 n·p 的 M·W 工作区；自动 ncv
    更大（nev = 12、p = 8：ncv = 64，基 72 列，p = 1 时 29 列）。每步 3 次块 M 作用（store_bv 2 次）。
  - driver 尚未开放 block_size（恒为 1）：30 个最近型测例上相同精度下几何平均加速 p = 2/4/8 为
    1.41×/1.36×/1.37×，但在小而易的问题上更慢，且 4–8 个测例的块运行残差底更高、追不上 p = 1；默认
    值的改变须先做 driver 端到端测量。预期用途是证书报缺副本后的收缩重跑（p ≥ 缺的副本数）。
- `krylov_schur_general`：欧氏内积，实数用实 Schur 形（dgees/dtrsen），共轭对永不拆开：
  m ∈ {nev, nev+1}，共轭对相邻、正虚部在前，nconv 不拆对，schur_form 无被截断的 2×2 块，
  前 nconv 列 Schur 基张成（近似）不变子空间。**不支持 deflation**（非正规算子的不变子空间的
  正交补不是不变的；用法上以更大的 nev/ncv 重跑代替）；block_size ≠ 1 抛 InvalidInput。
- 以上两者都通过 `history` 返回每次重启的收敛数与最大残差估计。
- **LOBPCG 与 Jacobi-Davidson**（`eigen_iterative.hpp`；直接作用于铅笔 (A, B)，不做谱变换）：
  - 共同约定：A 厄密、B（nullptr = I）厄密正定（不检查）；预条件子为近似 (A − sB)⁻¹ 的任意线性算子。
    **收敛按后向误差** η = ‖Ax − θBx‖₂ / ((norm_a + |θ| norm_b)‖x‖₂) ≤ tol 判定，即 driver 的验收
    标准（§4.5）；norm_a/norm_b 取自选项，为 0 时用 30 步 Lanczos 估计。返回前用新算的 A·X、B·X
    重算残差，`estimate` 就是真实 η。返回向量 **B-正交归一**。deflation 块 Y 须 B-正交归一
    （|YᴴBY − I| ≤ 1e-8，两者共用 `eigen_detail::require_b_orthonormal`）：迭代保持在 span(Y) 的
    B-正交补中，Y 不返回，残差是完整残差（Y 须准到约 tol）。
  - `lobpcg`（Knyazev；软锁定；Hetmaniuk–Lehoucq、Duersch 等的稳健基）：SmallestAlgebraic /
    LargestAlgebraic 的 nev 个；每轮把活动列的预条件残差 W 对 [Y X P] 及自身做 B-正交化（块 CGS2 +
    SVQB，丢弃数值相关方向），在 [X P W] 上用显式 Gram 矩阵做 Rayleigh–Ritz，P 在小空间内构造；
    Gram 矩阵病态时该轮丢弃 P（计入 restarts）。块大小自动为 min(n − d, nev + max(2, min(nev, 8)))。
    无预条件时**完全免分解**。
  - `jacobi_davidson`（Sleijpen–van der Vorst；JDQR 式锁定与厚重启）：目标 τ 附近的 nev 个（内部或
    外部）。**目标跟踪**：u 粗糙时修正方程以 τ 平移（配 M ≈ (A − τB)⁻¹ 相当于在 τ 处位移反演，V 按
    离 τ 的远近收集特征向量），‖r‖/‖Bu‖ ≤ 0.1 ×（到另一特征值的 Ritz 值的距离）后改以 θ 平移
    （局部快速收敛）。提取默认 Ritz（`JdExtraction::Auto`）；调和提取（Harmonic，带 Ritz 抛光和
    τ 处特征值的认证 Ritz 替换）**只用于 B = I**。内层：带预条件时右预条件柔性 GMRES（任意预条件子，
    斜投影），否则 MINRES（预条件子须 HPD）。起始空间 p = min(nev, max_basis − 1, n − d) 个随机向量，
    有预条件子时先做 `start_iterations`（默认 2）步子空间迭代 S ← orth(M B S)，把重特征值的多个副本
    一起带进 V（不做时 6/8/10 重特征值、nev = 6 的 24 次带预条件运行全部漏副本却报收敛，做了 0 次）。
    deflation 块视为已锁定，调用找 nev 个**新**对（证书后的收缩重跑）。预条件子只影响速度不影响
    精度（残差为双精度精确值），所以**单精度因子作预条件子即可得到双精度特征对**。
  - **JD 可能漏掉重特征值的某个副本**（无预条件子时 32 次中 28 次；以 converged = true 返回），本身
    无法察觉：必须由 driver 的惯性证书发现，并以已得向量作 deflation 重跑补齐。
  - 实测定位（`bench_lobpcg_jd_memory`，2 线程，k = 12，tol 1e-10）：JD + 单精度因子约为双精度位移
    反演 Lanczos 的 1.5 倍时间、一半因子内存，比「单精度因子 + 迭代修正」（driver 现有的内存回退）
    更快更稳；σ 距 6 重特征值 9.5e-9 / 2.5e-11（eps_s·cond > 1，单精度修正发散、双精度因子甚至奇异）
    时 JD 仍全部正确。LOBPCG 免分解路径在 syn-mem27-46 最小 12 个：15.1 s / 0.16 GiB（无预条件）、
    8.7 s / 0.54 GiB（单精度因子预条件），对照位移反演 Lanczos 双精度因子 12.7 s / 0.69 GiB。
    **driver 尚未调用二者**（P4：JD + 单精度因子作最近 k 个的内存受限回退；LOBPCG 用于极低内存的
    最小/最大 k 个）。

### 3.6 driver（`include/hsolve/solver.hpp`）
实现：`solver.cpp`（编排）、`steps.cpp`（共用步骤：位移、计数、分析与预算、自适应排序、代价模型）、
`definite.cpp`（HEP/GHEP：最近 k 个与窗口/谱切片）、`indefinite.cpp`（GHIEP）、`internal.hpp`。

0. **计时契约**：`PhaseClock` 把 `hs::solve` 的每一时刻记入且只记入一个阶段（嵌套作用域取最内层）：
   setup（输入检查、范数估计、问题类型判定、规划）、factor（所有符号分析——含自适应排序的 AMD
   探测——与所有数值分解——含惯性计数、B 正定性检测、A 的零主元计数）、iterate（Krylov 求解：
   回代、正交化、重启）、verify（Rayleigh–Ritz、残差、补救、选择、证书簿记）。因此
   `time_total_s == time_setup_s + time_factor_s + time_iterate_s + time_verify_s`，各项非负（P2 的
   time_verify_s 可能为负，已废）。证书的耗时分在 factor（端点计数）与 verify 中，报告不单列。
1. **诊断**：规模、nnz、实/复（A 复 B 实时 `to_complex`）、厄密缺陷（`hs::hermitian_defect`，超
   `hermitian_tol` 抛错，> 0 时取 `hermitian_part`）、‖A‖₂/‖B‖₂ 估计（~30 步 Lanczos 的最大
   |Ritz 值|：它是 ‖·‖₂ 的**下界**，从而 η 是真实后向误差的**上界**，验收偏保守）、
   scale = ‖A‖₂/‖B‖₂。文件输入已由 io 用缩放算法精确厄密化；内存输入（Python）只经过这里未缩放的
   缺陷，元素 ≲ 1e-154 或 ≳ 1e154 时会下溢/溢出（已知差距，`docs/PYTHON.md` §4、§7）。
   `threads > 0` 时 `omp_set_num_threads(threads)`（不恢复；Python 前端自己恢复）。
   B 正定性在符号分析之后判定：对角 B 直接看符号（免分解）；否则在主分析上分解 0·A + 1·B
   读惯性（主对象为双精度非 BLR；否则 `compute_inertia`）。
2. **策略**：
   - HEP/GHEP → 位移反演 + `lanczos_krylov_schur`（B 内积，block_size = 1）：nearest 用单位移
     （§3.6-8）；interval / smallest / largest 用窗口与谱切片（§3.6-8b）。
   - GHIEP → 位移反演 + `krylov_schur_general`，**只支持 nearest**（其余目标需要惯性计数，抛
     Unsupported）；零簇见 §3.6-9。
   - `method = lanczos`（免分解）抛 Unsupported；LOBPCG / JD 已在 eigen 层实现（§3.5），driver 尚未
     调用（P4）。
   - smallest / largest 的惯性搜索以谱两端的 Ritz 估计为种子：30 步 Lanczos（A；有 B 时为 B 内积下
     的 B⁻¹A，B 的分解复用主分析，对角 B 直接求逆）。
3. **内存预算**（显式公式）：分析前 factor_cap_MB = 0.95·(budget − current_rss − n·sizeof(T)·(ncv_g +
   6k_g))/1e6，k_g = k（interval：min(n, 64)），ncv_g = ncv（自动时 max(2(k_g + 16) + 1, k_g + 32)）；
   cap ≤ 0 即抛 OutOfMemoryBudget。`analyze` 后估计 ≤ cap 才做内存内分解，并把 cap 作为 ICNTL(23)
   传给 MUMPS（超限变成 INFO=−19 而不是被系统杀掉）；否则 `out_of_core ≠ off` 且外存估计 ≤ cap 时
   改用外存；仍超则抛 OutOfMemoryBudget 并报告估计值（绝不盲目分解）。BLR 与单精度只由配置开启
   （`blr = on`、`factor_precision = single`），driver **不**自动降级（P0 设计的「BLR → 单精度」降级链
   未实现）。
   窗口目标（§3.6-8b）在计数之后另查：n×m 输出加每片至多 4 对余量须 ≤ 0.9·(budget − 当前 RSS)，
   否则 OutOfMemoryBudget；一片的工作集约 7·c·n·sizeof(T)（基 ~2c，Rayleigh–Ritz 的 X/AX/BX 与乘积
   临时量 4c，选中块 c）限制片大小 c；**无固定特征值个数上限**（P2 的 max(2000, n/2) 已废）。
   已知未计入：GHIEP 后处理（结果向量、实子空间 W 及 AW、BW、投影解）约 88 B·n·m，wg-square-1e6
   （m = 19）上约 1.67 GB，对应预算项只有 ~0.86 GB；它决定该例的全程峰值（memio 实测，P4）。
3b. **自适应排序**（`ordering=-1`，默认）：先做 AMD 分析探测（极快），按代价模型预估嵌套剖分
   （SCOTCH）在本次运行的分解与回代上能省的时间，超过其额外分析代价（≈7×AMD 分析）才改用
   SCOTCH。模型：t_分解 ≈ 6e-7·n + flops/(12 GF/s·线程)，t_回代 ≈ 16 B·条目/12 GB/s；AMD 填充比
   >20 视为三维型（ND 保留 0.35 flops、0.6 条目），否则（0.6、0.85）；预计分解次数 = 1 + 2（证书）
   + 1（有 B），回代次数 max(40, 3k)。实测（8 线程）：二维波导 1e5 1.65→1.20 s，二维腔
   1.17→0.47 s；三维问题选 SCOTCH（比 METIS 分析快 35–50%）。
   注意：比较的是**实测**的 AMD 分析耗时，受机器负载影响——两次相同的 wg-square-1e6 运行分别选了
   SCOTCH（因子 7.24e7 条目、INFOG(21) 828 MB、峰值 2.92 GB）与 AMD（9.70e7、1140 MB、3.10 GB）
   （memio 实测）。需要逐位复现时固定 `ordering`。
4. **近奇异位移**（`factorize_near`）：`factorize` 返回 false（σ 在数值上落在特征值上）时**只扰动
   一次**：σ ← σ + 1e-8·max(scale, |σ|)（孤立特征值此时已在检测阈值之外约 1e3 倍；
   `max_shift_retries = 0` 时不扰动），记入报告。仍判奇异说明 σ 在比检测阈值更密的簇内，再往外移
   会改变「最近」的集合（P2 的第二次 10× 扰动在 syn-zero-cluster-real-1e5 上落到所需半径的 5 倍处，
   证书数出 166 个、找到 21 个）→ **在 σ 处关闭零主元检测分解**（以后在此位移的重分解照做）；再失败
   抛 Singular。「最近 k 个」的选择始终相对原始 σ。有意放在或紧挨重特征值的位移（§3.6-8b 的部分簇
   与重根重跑）被扰动时不告警、不计 shift_retries。
5. **迭代求解**：内层容差 = tol × inner_tol_factor；θ → λ = σ + 1/θ。nearest 恰好收敛 k 对
   （nev = k；完备性由证书负责，不靠多算余量）。主分解为单精度 / BLR 时算子带双精度迭代修正
   （≤ 10 步，rel_tol = 0.1·tol·inner_tol_factor）。
6. **精修**：HEP/GHEP 在原问题上做**秩揭示** Rayleigh–Ritz：B-Gram 矩阵 XᴴBX 做特征分解，丢弃小于
   最大特征值 1e-8 倍的方向（跨片去重时 1e-4），在剩余的 B-正交基上解投影铅笔。同一向量算两次、或
   补救把两列推到同一特征向量上，只留一份——「找到的对数」恒为不同 B-正交向量数，重复向量不可能
   凑满证书。GHIEP：所有运行的 Ritz 向量并集取实部与虚部张成实子空间（欧氏正交化，丢弃相关列），
   在其上解投影的不定铅笔（LU + Schur），一次精修全部候选。
7. **残差验证**：每对计算 ‖Ax − λBx‖₂、相对残差、后向误差 η（§4.5）。未达标者做残差修正迭代
   x ← x − K̃⁻¹(A − ρB)x（ρ 为 Rayleigh 商；对任意近似分解 K̃，其不动点都是 (A,B) 的精确特征对；
   每对至多 3 步，K(σ) 的分解按需重建），之后再做一次秩揭示 Rayleigh–Ritz；仍不达标则
   accepted=false。η > max(1e-6, tol) 的对视为幽灵，不计入找到数。
8. **惯性证书（nearest，HEP/GHEP；复数经实嵌入同样可用）**：
   - 窗口：r = max|λᵢ − σ|，a ≤ σ − r、b ≥ σ + r，各自推到与更外侧最近的 Ritz 提示值（ritz_all 的
     映射值）的中点；无提示时外推 max(1e-3·|端点 − σ|, 1e-8·scale)。
   - 通过条件 count_inertia == count_found（count_found 为窗口内不同 B-正交向量数）。少 → 以已收敛对
     （η ≤ 10·tol）作 deflation、在 σ 处重分解后重跑（nev = 缺数 + 2），至多 2 轮；多 → 告警
     （重复/幽灵收敛）。
   - `count_below(s)` = ν₋(A − sB)：先带零主元检测计数；被判奇异时，s 若是谱隙点（证书端点）就在
     s 处关闭检测计数（稠密簇内任何移动都会越过特征值；向后稳定的 LDLᵀ 对离 s ≫ eps·‖K‖ 的特征值
     计数精确）；s 若是任意点（用户区间端点、插值切点、零带边界——可能恰在一个（重）特征值上）就在
     s ± step/10 处关闭检测计数（窗口用 step = 1e-10·max(scale, |s|)，即偏移 1e-11 相对）。仍奇异则
     向外移 step·4ⁱ（至多 8 次，计入 endpoint_moves）。主分解为单精度或 BLR 时改用 `compute_inertia`
     （双精度、无 BLR）。
   - **目前复用主分解对象**：在端点 factorize(K(a))、factorize(K(b)) 读惯性，重跑或补救前再在 σ 处
     重分解（每轮证书 2–3 次分解；三个 3-D 测例 2 线程实测占总时间 23–37%）。InertiaCounter
     （§3.4）的接入在 P4 进行中，计划规则（certpar 实测）：仅当主分解为单精度/BLR、排序为嵌套剖分、或 AMD 分析耗时
     < 0.3 × 主分解耗时时用计数器（低填充输入上计数器的分析反而更贵）；主因子 INFOG(21) + 计数器
     估计超出上限时先 `release_factors()`；主因子完好时重跑不再在 σ 重分解；B 正定性检测也可改成
     计数器的 count(0, 1)；可选的单精度筛查计数只用于 B = I 或对角 B，且位移须离所有特征值
     > 1e-4·κ(B)·(scale + |s|)。
   - GHIEP 无惯性证书（报告中注明）；P0 设计中「换种子、更大 nev 重跑」的延迟重启检查未实现。
8b. **窗口与谱切片**（interval / smallest / largest，B 正定或 B = I；以及零带干扰的 nearest，见 9）：
   - 窗口端点与所有切点都是同一分析上的精确 Sylvester 计数，每片都知道自己含几个特征值；各片单独
     求解、单独认证，最后合并。
   - 片大小：c_target = `slice_size`（> 0）或代价模型最优值（钳到 [16, 1000]），c_max = `slice_size`
     或 1.25·c_target。代价模型（`steps.cpp`，在 yee-pec-box 1e4/1e5 的切片扫描上标定）：每对回代
     2.1 + 33/c 次、基向量遍历 200 + 7c 次（每线程 6 GB/s、上限 60 GB/s）、每片 2.2 次分解，分解/回代
     时间沿用 §3.6-3b 的模型；最优 c* = √((2.2·t_分解 + 33·t_回代)/(7·向量字节/带宽))。只需落在实测
     平坦最优区内（同一 1e5 因子上回代时间在共享机器上波动 69–98 ms）。
   - 切点从窗口下端向上推进，每片约 c_target 个、至多 c_max 个（剩余 ≤ 2c_target 时对半分）。计数
     搜索 `find_count`：以上一片的局部密度（否则窗口剩余部分的平均密度）给首猜，再做带 Illinois
     修正的计数插值（regula falsi，步子保持在括号内 5%；某次计数等于括号端点的计数时改二分）；开放
     端从越界的那一点几何扩张。计数分不开的簇（同一对端点计数的括号缩到首次出现时的 1/32，或宽度
     1e-8·max(scale, |x|, |y|)）终止搜索，返回簇旁已认证谱隙的中点（最大 k 取簇下方，其余取上方），
     绝不返回贴着簇的点。
   - 位移：内部片取中心（LargestMagnitude）；片外一侧没有特征值（谱端）或另一端紧邻零带时，在片端
     用单边位移（SmallestAlgebraic：取位移以下；LargestAlgebraic：取以上），直接复用该计数点的分解；
     被判奇异的端点（其上有特征值，如用户端点）不作锚点——有限片改取中心，开放片取该端旁谱隙中点。
   - 每片收敛恰好其计数个对；离切点 δ = 1e-9·max(scale, |cut|) 以内的片外对也保留（这么近的特征值
     可能算到另一侧），交给去重。缺对时以已得对作 deflation 重跑，只要还有新对（至多 缺数 + 4 轮、
     上限 32 轮、连续 2 轮无新对即停）；已得对中有重特征值时，重跑位移放在离它 1% 谱隙处（重根主导
     算子约 100 倍，舍入补出其余副本；正放在重根上会把数值重簇拆成逐个收敛的 θ）。
   - 合并：来自不同片、特征值相差 ≤ 2e-9 相对的对成组，以 B-Gram 秩（1e-4）+ Rayleigh–Ritz 去重；
     全局证书逐窗口要求「不同对数 == 计数」。
   - smallest / largest：窗口 (−∞, U] 与 [L, +∞)，U（L）由上述惯性搜索确定：要求 ν(U) ∈ [k, k + slack]，
     slack = t_分解 / 边际每对代价（钳到 [1, max(16, k)]：多算几对比再做一次计数便宜时停止细化），
     再在 U（L）处单边运行。第 k 个特征值落在数值相等的大簇（超出 k 的副本多于 max(16, k)，如零空间）
     中时为「部分簇」窗口：只求仍需的副本数，由在其均值 ± Δ（Δ = max(16 × 副本散布, 1e-9 相对)，
     ≤ 1e-8 相对）的两次计数证实整簇数值相等后接受任意这些副本；否则整簇求解（先查内存预算）。
   - 报告：每次运行一条 `slices[]`（窗口、位移、which、计数、找到数、nev、重跑、重启、算子作用
     次数、是否通过）；`sigma_used` 为首片位移。
9. **零模（零空间）处理**：EM 零空间可达 n 的 20% 以上，**不得显式 deflation 或剔除重算**；零簇
   靠位移放置排除，不靠事后过滤。`zero_filter_tol` 定义为相对 scale（< 0 关闭）。
   - HEP/GHEP（仅当 `zero_filter_tol` ≥ 0；带宽 band = zero_filter_tol·scale）：在 −band、+band 两点
     计数（零带边界），窗口在这两个计数点之间切掉零带——过滤与计数完全一致，`zero_modes_filtered`
     即带内特征值数（由惯性得出）；紧邻零带的片把单边位移放在远离零带的一端；一直延伸到零带的开放
     窗口改在谱端之外的点封口。nearest：先做普通单位移运行；若零带进入选择或证书窗口，转交窗口求解
     [σ − r, σ + r] 减去零带——r 取第 k 个与下一个不同 Ritz 提示的距离中点（无下一个时 1.25 × 第 k 个），
     再按计数扩大到窗口在带外含 k 个特征值（单位移在零带附近会收敛零空间副本）。未设
     `zero_filter_tol` 时零空间特征值就是普通特征值（smallest / largest 可走部分簇规则）。
   - GHIEP（零簇从不被需要；带宽 max(zero_filter_tol, 1e-8)·scale）：(1) σ 处普通运行，其 k 个最近
     对不触及零带则保留；(2) 否则（或 σ 在零带内）在主分析上分解 A 数零主元得零簇维数
     （`zero_cluster_dim`），有簇（或用户设了零带）时改用两侧单边运行 σ₊ > band（LargestAlgebraic，
     θ>0 ⇔ λ>σ₊）与 σ₋ < −band（SmallestAlgebraic），零簇落在不需要的一端。每侧位移放在该侧已持有
     的最外特征值（零簇，或普通运行的最外对）与下一个不同特征值的**中点**；后者由内缘处一次探测运行
     （2 对、容差 1e-6、≤ 4 次重启）找到——位移贴着零带边缘时零簇的 |θ| = 1/band 压过所需值，残差
     停在 1e-10..1e-9，中点处两侧邻居平衡。探测无该侧 Ritz 值 → 该侧为空；所需值符号不对 → 该侧耗尽。
     实部位于两侧位移之间的复特征值对两侧都不需要：由各 Krylov 空间里出现两次的复 Ritz 估计找出，
     在其实部的 1/4 处做一次单边恢复运行（`recover_strip`），未决时告警且 success=false。全部 Ritz
     向量一次性在投影铅笔上精修，选离 σ 最近的 k 个带外特征值；某侧收敛范围未覆盖最终半径时加大
     nev 重跑该侧。
10. **GHIEP 实/复分类**：|Im λ| > complex_tol·max(|λ|, scale) **且** 实投影对未通过后向误差测试而
    复对通过，才算复特征值；复特征对单独输出（`complex_eigenvalues`），实特征对不受影响。
11. **输出**：见 `docs/FORMATS.md §3`（含 `slices`、`zero_cluster_dim`、`zero_modes_filtered` 与
    分阶段计时）。

### 3.7 util（`include/hsolve/log.hpp`、`include/hsolve/util.hpp`）
- `Logger`：JSON lines 写文件 + 人类可读 stderr；事件名小写下划线，字段扁平。
- `Timer`、`peak_rss_bytes()`、`current_rss_bytes()`、`max_threads()`、`build_info()`。

## 4. 数值约定

### 4.1 位移反演
K(σ) = A − σB，OP = K⁻¹B。A x = λ B x ⇔ OP x = θ x，θ = 1/(λ − σ)。离 σ 越近 |θ| 越大。

### 4.2 惯性（Sylvester）
B 正定时，ν₋(A − sB) = 「小于 s 的特征值个数」，由 LDLᵀ 负主元数得到（valid 时）。区间 (a, b)
内特征值数 = ν₋(A − bB) − ν₋(A − aB)。端点须落在谱隙中，否则数值惯性不可靠（零主元检测可能漏掉
恰在端点上的特征值，它会被数到任意一侧；任意点的计数规则见 §3.6-8 `count_below`）。

### 4.3 GHIEP
B 不定时惯性计数不成立，理论上可有复特征值。用非厄密 Krylov-Schur；实/复分类见 §3.6-10。

### 4.4 收敛（内层）
Krylov 求解器：Ritz 估计 ≤ tol_inner·|θ|（θ = 0 时用最大 |Ritz 值|）；块模式在块边界也检查
（§3.5）。LOBPCG / JD 直接以后向误差 η ≤ tol 判收敛（§3.5、§4.5）。

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
- **禁止**：稀疏矩阵稠密化（唯一例外：测试/评测代码中 n ≤ 4000 的参考解）；全局可变状态（唯一
  例外：串行化 MUMPS 调用的进程级锁，§3.4）；无种子随机数；库代码中 `std::cout`/`printf`。
- 测试：`tests/cpp/test_<module>*.cpp`（GoogleTest），**套件名以模块名开头**（`Sparse*`、`Io*`、
  `Linalg*`、`Factor*`、`Ops*`、`Lanczos*`、`KrylovSchurGeneral*`、`Lobpcg*`、`JacobiDavidson*`、
  `Driver*`；P3 新增如 `LinalgDenseOpsBlockCgs`、`FactorInertiaCounter`、`LanczosBlockTyped`、
  `LanczosBitwiseTyped`、`LobpcgTyped`、`JacobiDavidsonTyped`、`DriverSlicing`、`DriverZeroCluster`）；ctest 名为
  `<可执行文件>.<套件>.<用例>`，`tools/dev/test.sh release test_factor` 只跑该文件，匹配不到测试即
  报错。共享夹具在 `tests/cpp/support/`（`hsolve_test_support`），只追加不改签名。
  数值测试尽量用有解析解的问题。Python 工具与 Python 接口的测试在 `tests/py/`（pytest；
  `test_python_binding.py` 默认导入 `build/release/python`，`HSOLVE_PYTHON_DIR` 可改）。
- 测量程序：`bench/<name>.cpp`（目标 `bench_<name>`，见 `bench/README.md`），提交信息与文档中引用的
  数字都应能由其中某个程序经 guard.py 复现。
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
| P3 | **已完成**：低内存输入装配与无映射铅笔；确定性正交化（融合 CGS2、DGKS、块 CGS2）；MUMPS 约定补充与 InertiaCounter；块 Krylov-Schur（eigen 层）；LOBPCG / JD；谱切片与惯性搜索的单边目标、零带切除、GHIEP 零簇移位放置；互斥分阶段计时；Python 接口；评测工具（汇总、合并参考、scipy 基线）。**未做**：SpMV SIMD；GHIEP 重特征值的块 Krylov（一般 Krylov-Schur 仍只支持 p = 1） |
| P4 | driver 集成：InertiaCounter 证书、JD + 单精度因子的内存受限回退（需 ops 的只回代算子）、block_size 端到端测量后再定默认、GHIEP 后处理内存与预算、确定性的排序决策；故障场景验证矩阵、方案文档、复现材料 |
