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

v5（P5c 集成：p5c-integration 1bd75f9 加集成修正，至 9201c2c）——src/verify 接入 driver：nearest 的完备性检验
`certificate = exact | randomized | auto | off`（单精度计数先经双精度复核、auto 选单精度计数要求 10 倍余量、
随机检验只在 κ(B) 有证明时出结论、选中对少于 k 时也重跑，§3.6-8）；输入检查（缩放的厄密缺陷、拒绝零矩阵与
非有限值、恢复线程数）与 B 的正定性检验链（分析之前的廉价检验、IC + Gershgorin 严格证明及其预算，§3.6-1）；
自适应排序改为 16 线程标定的代价模型（候选 AMD / SCOTCH / 自研嵌套剖分 / METIS / PORD，config 可直接用
`ordering = 10`，§3.6-3b）；σ₀ 处按零主元数识别稠密簇（§3.6-4）；块宽规则 `recommended_block_size` 与块步的
单投影快路径 `try_fast`（§3.5），driver 的 nearest 运行用该规则（从 4 加宽到 8 须有 10% 的模型收益；内存只预留
P4 块宽的基，§3.6-3、§3.6-5）；收缩重跑在缺的对收敛后即结束（`KrylovSchurOptions::lead_nev`），自动块宽之后的
重跑至多用 P4 块宽（§3.6-8）；certifier 的 InertiaCounter 推迟其双精度实例的分析（§3.4）；GHIEP 的分析前内存
预留按峰值列计、报告实际块宽；新模块 src/blas（BLAS hook，默认链入 hsolve、route all，§3.8，实测见
`docs/BLAS.md`）。

v6（P8「certcost」，分支 p8-certcost）——证书的分解失败不再结束求解：计数源（主对象 / InertiaCounter）失败时换另一个，
估计放不下预算余量的计数器不尝试，都不能计数时 nearest 返回已收敛的对、证书「已做但未完成」（§3.4、§3.6-8，
`certificate_fault_count` 故障注入）；一般 HPD B 的 κ 与 λ_min(B) 由 IC + Gershgorin 证明（`prove_condition_bound`），
随机检验与单精度计数因此也能在一般 B 上认证（§3.5b、§3.6-8；只影响 auto / randomized，默认仍为 exact）；更便宜的
精确计数（并发端点计数 8 + 8 线程、一次分解两端、计数设置、计数来源）实测为负结果，未采用（§3.4）；MUMPS 的 SCOTCH
排序在多线程下仅靠随机种子不能复现（§3.4）。实测见 `results/p8_certcost.md`。评审修正（p8-certcost-fix）：主对象替补
计数器时也先查计数余量（P8 初版在小例上超预算 38%）；窗口目标从第一片起只放弃计数不可能的那一片、返回其余片的对；
「无法计数」的错误码按失败种类定；故障注入增加失败种类（其它 MUMPS 错误、`std::bad_alloc`）与「从第一片起」位，
位 8 也作用于随机检验前的重分解（§3.6-8）。

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
src/driver   诊断、策略、位移管理、谱切片与单边目标、零簇处理、完备性检验（惯性证书与 certificate 模式）、
        │    精修、残差验证、报告(JSON)
src/verify   完备性检验构件：随机排除检验、单精度计数规则、代价模型（verify.hpp；依赖 eigen、factor、
        │    sparse、linalg；driver 的 nearest 证书经 src/driver/certificate.* 调用，§3.6-8）
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

src/blas     （P5，不属于 libhsolve）BLAS hook：MUMPS 与 hsolve 发出的稠密 BLAS 调用按形状改道到窄核 /
             并行切分，其余原样转发真实 BLAS；默认链入 hsolve 可执行文件，或以 libhsolve_blashook.so
             预加载（§3.8，docs/BLAS.md）
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
  块 Krylov-Schur 专用；一般 Krylov-Schur 块模式的回退核；P5c 起另有单投影快路径 `try_fast`，§3.5）与
  `eigen_detail::block_cgs_cholqr2`
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
    `omp_get_max_threads()`）。逐位复现还需可复现的排序（driver 的 `deterministic = true` 自动满足：
    沿用 P4 的排序规则，AMD 或固定种子的 METIS，§3.6-3b；显式 `ordering = 10`（自研嵌套剖分，排列与线程数
    无关）也可复现；显式 SCOTCH 不可复现，见下）。
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
  - **SCOTCH 排序不可复现**：同一进程内两次分析得到不同排序（flops 差 10%），不同进程间也不同
    （wg-square-1e6 固定 ordering=3，4 次运行因子条目 7.202e7–7.275e7）；42 个数据测例上同一进程的
    3 次 SCOTCH 分析估计在每个测例上都有变化，METIS（固定种子）在任何测例上都没有（1888a72，
    `bench_drvint_fix_ordering`）。`FactorOptions::deterministic` 只覆盖分解与回代；driver 的
    `deterministic = true` 因此在自适应排序中改用 METIS（§3.6-3b）。
- 错误约定：工作区不足（−8,−9,−14,−15,−17,−20），以及工作区预留低于 cap 时的 −19，由 `factorize`
  内部重试 ≤3 次（ICNTL(14) 翻倍、预留随之增长，最后一次给满 cap；计入 `workspace_retries`）后抛
  FactorizationFailed；−13/−19（分配失败 / 预留已是 cap 仍超）抛 OutOfMemoryBudget；其它负 INFO 抛
  FactorizationFailed（消息含 INFO(1)/INFO(2)）。
- **工作区预留**（P7 评审修正）：ICNTL(23) > 0 时 MUMPS 按 ICNTL(23) 而不是按估计确定工作区，地址空间
  随之增长上限的一半到大部分（虚拟预留，常驻只增 INFOG(21)；syn-mem27-46 16 线程、cap 10125 MB：VmPeak
  10.1 GiB 而峰值 RSS 0.85 GiB）。RLIMIT_AS（`ulimit -v`）把预留全额计入，P7 时 cap 取到上限的约 0.8
  就失败（INFO −13，或 OpenBLAS 分配不到缓冲而段错误）。所以 `factorize` 给 MUMPS 的 ICNTL(23) 是
  min(cap, (1 + ICNTL(14)/100)·估计 + 32 MB)——内存内用 INFOG(16)、外存用 INFOG(26)，按估计定额，
  cap 只作上限；重试时随 ICNTL(14) 增长，最后一次给满 cap（`FactorStats::workspace_reserved_mb`；
  InertiaCounter 的计数实例同样按其自己的估计）。
- `scaling` 默认 ICNTL(8)=7（每次分解重算缩放），使一次分析服务不同位移。
- `FactorEstimate`：INFOG(20)（因子条目估计）、INFOG(16/17) 内存、INFOG(26/27) 外存、
  INFOG(36/38) BLR 估计（仅当分析时启用 BLR，否则 −1）、RINFOG(1) flops。
- 单精度 / BLR 分解的 inertia() 是舍入/压缩后矩阵的惯性，**证书不得使用**：driver 在主分解为
  单精度或 BLR 时一律在 InertiaCounter（双精度）上计数（§3.6-8）。
- `compute_inertia(pencil, α, β)`：自建对象（自做一次分析）、强制双精度且无 BLR、丢弃因子
  （ICNTL(31)，只占工作内存）、奇异时返回 valid=false 不抛；driver 只在主分解为单精度/BLR 时用于
  B 正定性检测与 GHIEP 的 A 零主元计数，另用于测试。B 为 HPD 当且仅当 `valid && neg==0 && zero==0`。
- `release_factors()`（非纯虚，默认返回 false；MUMPS：JOB=−4）：释放数值因子与工作区、保留分析。
  之后 `solve` 抛 Internal、`inertia().valid = false`（valid ⇔ solve 可用，ShiftInvertOperator 与
  driver 都以它判断），计数与 `stats()` 仍描述被释放的那次分解——需要时先拷贝 `inertia()`。
- `InertiaCounter<T>(analysed_source, options)`：在已分析主对象的模式上计 ν(αA + βB)，**不动主对象**
  （主因子保持有效，证书后不必在工作位移重分解）。机制：自有的只计惯性 MUMPS 实例（ICNTL(31)=1），
  用主对象的主元顺序分析（PERM_IN，不再做一次填充排序）；`count(α, β, detection, precision)`、
  `count_pair(…)`（两次依次计数——并发不可行，见上；diag(K(b), −K(a)) 一次分解慢 1–10%、内存 1.5×）、
  `estimate()`、`stats()`。双精度实例默认在构造时分析；第三个参数 `defer_double_analysis = true`（P5c 集成）
  把它推迟到首次使用（第一次双精度计数或 `estimate()`），只做单精度计数的计数器因此只付一次分析（单精度实例
  的，在首次单精度计数时；n ~ 1e5、16 线程约 0.06 s），推迟的分析出错时也在那次使用时抛出；
  `double_analysed()` 告诉双精度实例是否已分析。实测（2 线程，主因子常驻，driver 路径 = 在两端点重分解
  主对象，5 次交错中位数）：3-D SCOTCH 测例证书对 −12..−15%（syn-mem27-46 3.85 → 0.06 + 3.27 s），
  2-D AMD 测例 −2%，低填充的复数实嵌入（syn-near-singular-complex-1e5）+42%（计数器的分析 0.48 s 与 AMD 分析相当）。
  一次计数的工作内存 INFOG(21) 为主因子的 39–59%（例 289 vs 683 MB）；主因子常驻时峰值 RSS 增加
  9–228 MB，先 `release_factors()` 则不增加。
  精度：双精度计数对距位移远于 O(eps·‖K‖·增长因子) 的特征值精确；零主元检测通常标出更近的位移，
  但它是同一个依赖消去顺序的启发式，可能漏检（计数器按自己的顺序消去，漏在与主对象不同的位移上：
  2-D Laplacian 24 个精确特征值位移中漏 1 个，漏检的计数落在 [#below, #below + 重数] 内），故
  valid == true 只在位移位于谱隙时才是证书——端点仍须放在谱隙中点。单精度计数（smumps；3-D 测例快
  45–53%）是被 O(eps_s·‖K‖) 扰动矩阵的精确计数：距最近特征值 δ·(scale + |s|) 内会数错（实测到
  δ ≈ 1.6e-6，稠密复谱 5e-6；δ ≥ 1e-5 从未出错），单独只能作筛查值。
- **driver 的用法**（P4，§3.6-8 `CountSource`）：nearest 证书在模型判定计数器划算时
  （`counter_pays`）、以及主分解为单精度/BLR 时在 InertiaCounter 上计数，每次计数前释放主因子；
  窗口/切片默认在主对象上计数（计数点兼作单边位移），只在主因子 INFOG(21) + 计数器估计超过上限时
  释放。4 线程端到端实测：证书时间比主对象计数少 0.05–0.14 s（5–13%），峰值 RSS 低 2–7%
  （667d8e9）。P5 起 `certificate = auto` 在 §3.5b 的规则带 10 倍余量成立时用单精度计数（证书方法
  exact_single）：它们在 certifier 自建的 InertiaCounter 上做，可疑时在同一计数器上以双精度复核，同一
  计数器随后也服务该轮的精确计数（不再建第二个计数器、也不为它再分析一次；§3.6-8）。这个计数器推迟其双精度
  实例（`defer_double_analysis`，P5c 集成）：单精度计数一次通过的证书只付单精度实例的一次分析（之前还有一次
  用不到的双精度分析，yee-pec-box-1e5 约 0.075 s）；双精度实例在复核或精确计数第一次用到它时才分析。certifier
  的 `inertia_counter` 事件因此不再报 incore_mb（读它会触发分析），`single_counts` 记下计数器已做的分析次数。
  上面 P4 用法的计数器（`CountSource`）只做双精度计数，照旧在构造时分析。
- **P8：计数的内存与失败**：`InertiaCounter::estimate(precision)` 给出某精度计数的工作内存（单精度时先分析单精度
  实例），driver 在计数前据此判断能否放进预算（§3.6-8 计数余量）。计数在 ICNTL(23) 上限下的 MUMPS 失败（INFO −19，
  `OutOfMemoryBudget`）照旧抛出，由 driver 接住换计数来源（§3.6-8）。
- **P8：更便宜的精确计数——实测为负结果**（16 线程测量槽，gpu01；`bench/certcost_concurrent.cpp`、
  `bench_certpar_inertia --exp blockpair`、`tools/bench/certcost_concurrent.sh`，`results/p8_certcost.md` §3）。nearest
  证书的两次精确计数 ν(a)、ν(b) 是同一主元顺序上相互独立的只计惯性分解，最多能做的是：
  - **并发（8 + 8 线程）**：进程内不可行（上文：MUMPS 实例间不线程安全；全局锁串行化一切 MUMPS 调用）。上界用两个
    进程测：两个 8 线程计数同时跑（各绑槽内 8 核）对顺序的 16 线程计数对只快 0.99–1.29×——ss-qa8fk-qa8fm 1.29×
    （0.325 → 0.253 s）、cav2d-rect-pert-1e5 1.26×（0.052 → 0.041 s）、三维测例 0.99–1.12×（计数受内存带宽限制：单个
    8 线程计数 0.42–2.58 s，两个同时跑时各 0.63–4.33 s）；1e4 小测例 ~2×，但每对只 5–10 ms。即使实现了，整次求解在
    qa8 上至多快 ~4%、cav2d 上 ~3.6%。进程内唯一的隔离办法（fork 出子进程计一端）不可用：OpenMP 用过之后 fork 的
    子进程在 MUMPS 里 SIGSEGV（8 例中 6 例每次都是）；dlmopen 第二份 MUMPS（独立 Fortran 模块状态）在 glibc 2.17、
    重复的 libgomp / OpenBLAS 下风险高、收益以上述上界为限，未做。
  - **一次分解两端**（diag(K(b), −K(a))，MUMPS 自己的 L0-OpenMP 并行两棵独立的树）：16 线程下为顺序对的
    0.96–1.09×，还要自己的分析（0.04–0.28 s）与 1.8–2.5 倍工作内存——更慢。
  - **计数的设置**：不缩放（ICNTL(8) = 0）或不做零主元检测只省 ≤ 3%（噪声内）；计数线程 16 在每个测例上都快于 8。
  - **计数来源**：在主对象上重分解还是在 InertiaCounter 上计数——`inertia_counter = on` 对现行规则（auto）在 31 个
    HEP/GHEP 测例上几何平均慢 3.4%（ABBA 3 次），规则已经选对；与其它工作重叠：计数之前只有 verify（~0.01 s）不用
    MUMPS，Krylov 的回代与计数抢同一把锁，无可重叠之处。
  结论：在 MUMPS 5.8.2 上精确证书的代价就是两次分解，默认（exact）保持现状；ss-qa8fk-qa8fm / cav2d-rect-pert-1e5
  对调优 SLEPc 的差距不能靠更便宜的精确计数消除。
- **P8：SCOTCH 排序的可复现性**（MUMPS 估计逐次波动的来源）：MUMPS 里的 SCOTCH 排序随 OpenMP 线程数多线程运行；
  2 个及以上线程时即使 `SCOTCH_randomSeed` + `SCOTCH_randomReset` 也逐次不同（syn-lap3d-aniso-1e5，8 线程：因子
  条目 33.1e6–35.2e6，外存估计 208–272 MB），单线程时跨进程复现（同一进程内每次分析前 reset 也复现），但分析慢
  2.6–3.7 倍（四个 1e5 测例 1.4–2.6 s 对 0.45–0.8 s）。不能廉价地复现；driver 改为不依赖估计的精确值（§3.6-8 的
  计数余量与失败回退）。
- **自研嵌套剖分**（P4，`compute_ordering`、`OrderingOptions`、`src/factor/ordering*.cpp`，无新依赖）：
  - 动机：MUMPS 内的排序是串行的（METIS/SCOTCH 占其分析的 90–97%；SCOTCH_PTHREAD_NUMBER 无效）；
    METIS/SCOTCH 不能直接链接（只是 libdmumps 的间接依赖，ld 拒绝间接 DSO，dlopen 需要 glibc 2.28
    sysroot 链接行上没有的 libdl）；METIS 5.1 用 libc rand()，并发调用不可复现。
  - 算法：METIS_NodeND 式多级顶点分隔子（有序重边匹配、贪心生长、单侧节点 FM + 随机打破平局），
    大图取 4 倍粗化图上 5 次试分的最优，叶子（≤ `leaf_size`，默认 120，上限 2048）用精确最小度；
    每个分隔子的两半与 5 次试分都是 OpenMP 任务（粒度 512 顶点）。随机数为按子图播种的计数器式
    生成器 ⇒ **排列与线程数无关，逐位相同**。`imbalance` ∈ [1, 1.5]（默认 1.2；1.5 在三维实例上
    flops +14..+70%，更大时退化），越界抛 InvalidInput。总是开自己的并行区并等待全部任务，可在
    并行区内调用（无嵌套并行时单线程、结果相同）。
  - 复数铅笔：至少一半的非对角耦合为复数时排 K 的 n 顶点图并把每个顶点的两个嵌入变量相继消去
    （Compressed），否则排 2n 顶点的嵌入图（Full；`OrderingOptions::embedding` 可强制）。
  - 接入 MUMPS：`FactorOptions::ordering = kOrderingNestedDissection`（10）时 analyze 先算排列，以
    PERM_IN（ICNTL(7)=1）交给 MUMPS；`ordering_threads`（0 = `threads`）为排序线程数；
    `FactorStats::ordering_s` 是 analyze_s 中排序所占部分。同一分析给定排列的部分只需 0.03–0.08 s
    （n ≈ 1e5）、0.5–0.6 s（n = 1e6）。
  - `FactorOptions::block_analysis`（1 开、0 关（默认）、−1 自动：至少一半耦合为复数时开）：复数时
    以 ICNTL(15)=1 把嵌入的 {i, n+i} 声明为块，MUMPS 自己的排序便排 n 顶点图（全复数测例 SCOTCH
    分析 3.0 → 0.30 s；Bloch Yee（3% 复耦合）METIS flops 2.24e12 → 2.69e12，故默认关）；
    `FactorStats::block_analysis` 记录是否生效。
  - 实测（2 线程，`results/p4_measurements.md` §2）：分析比 MUMPS 串行 METIS 快 1.3–1.6×（串行时
    慢 1.2–1.35×），flops 在 METIS 的 −7..+12% 内（16 线程的实测见下）。
    按 driver 实付（分析 + 3 次分解 + 回代）计，yee-pec-box-1e5 上 2 线程打平，二维测例 AMD 最好。
  - **driver 的用法**（P5）：config `ordering = 10` 直接使用它；自适应排序（`ordering = −1`）把它列为
    候选之一，由 16 线程代价模型选择（§3.6-3b）。16 线程实测（`bench/anaint_orderings.cpp`，测量槽）：
    分析的串行比例约 0.38（4 线程时 0.19；16 线程的分析只比 4 线程快 1.14 倍，29 个测例中位数）；高填充
    三维测例（Yee 网格、ss-ge87h76、ss-tem152078）上 SCOTCH 的因子只需它的 0.80–0.95 倍 flops，因此候选
    顺序中 SCOTCH 在它之前。分隔子质量与排序的并行扩展是 factor 模块的待办（ss-tem152078 上它的排序
    1.24 s，SCOTCH 的整个分析 0.73 s）。

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
  停滞或发散（实测 syn-mem27-46、σ = 0：10 步后 ‖r‖/‖Bx‖ ≈ 5e-6）。`last_apply_converged()`（P7）
  报告上一次 apply 的每列是否以「达到 rel_tol 或舍入水平」结束（false：某列因修正被拒或步数用尽而
  停下，返回的是扰动矩阵的算子）。driver 用 `DriverShiftInvert` 包装它：非精确因子的每次 apply 都
  检查，未收敛且相对残差仍高于 Krylov 容差 tol·inner_tol_factor（修正目标的 10 倍：在可达精度处停滞
  只是略差，wg-square-1e5 1.08e-12 对 1e-12；eps·cond(K) ≥ 1 时残差停在 1e-5 以上，syn-mem27-46
  2.4e-5）即放弃该次尝试、改用双精度因子重解（§3.6-3「非精确因子」）。修正工作区 2·n·p 标量。
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
- **`KrylovSchurOptions::lead_nev`**（P5c 集成，单向量与块模式都有；`krylov_schur_general` 对非零值抛
  InvalidInput）：0 < lead_nev < nev 时，按 which 排名最前的 lead_nev 个 Ritz 对全部收敛即结束运行（单向量在
  轮末、块模式也在提前停止探测处检查），不论其余 nev − lead_nev 个想要的对如何；它们照样决定基数与保留集，
  按其估计返回（收敛与否；结果契约不变，converged == (nconv ≥ nev)）。用于调用方多要的余量对：宽于 tol 的簇
  内的对永不收敛（如 Maxwell 铅笔的 curl-curl 零空间，θ 上只在约 1e-10 相对内为零），不能让它把运行拖到
  max_restarts。0（默认）或 nev：nev 个都须收敛；[0, nev] 之外抛 InvalidInput。
- **块模式**（`KrylovSchurOptions::block_size = p > 1`，块 Krylov-Schur，Zhou & Saad 2008）：每步
  一次 p 列的 `op.apply`（多右端回代）；新块对 [D V] 做块 CGS2、块内做 M 内积 Cholesky-QR
  （`BlockOrthonormalizer`：G 非正定、R 的列尺度条件数 > 1e5 或某列在两遍后仍相关时回退逐列 CGS2，
  相关列换成确定性随机向量——块版的 Lanczos 中断）；投影矩阵为块三对角加重启的箭形块，估计为
  ‖R s_last‖₂。起始块：`initial` 的前 min(p, cols) 列 + 随机列。
  - ncv：基数 = 单向量的 ncv（自动或给定）上取整到 p 的倍数，且 ≥ s + 2p（s = nev + 1 上取整到 p 的
    倍数）；给定 ncv 时即为基数，自动时再加 4p（受 n − d ≥ 2(ncv + p) 限制）。每次重启保留
    k = nev + (ncv − nev)/2 下取整到 p 的倍数，至少 s（留一个备用 Ritz 对）、至多 ncv − p。
  - **提前停止探测**：在块边界也做 Rayleigh–Ritz（第一轮从基数处起，之后从 k + p 起），nev 个全部
    （或 `lead_nev` 个最前者，见上）收敛即结束本轮。块运行因此恰好达到 tol，而 p = 1 只在轮末检查、
    通常超出 tol 几个量级——两者须在相同残差下比较，不能在相同 tol 下比较。
  - 自由空间不足（n − d < 2(ncv + p)），或 p 大到放不下一个块基，则用一轮覆盖整个自由空间的单向量
    循环精确求解（n − d 次作用）。
  - 重数 ≤ p 的特征值的全部副本一轮内找到；超出 p 的副本只能靠舍入进入（同 p = 1）。
  - 内存：基 n·(d + ncv + p) 标量（store_bv 时加倍），无 store_bv 时另加 n·p 的 M·W 工作区；自动 ncv
    更大（nev = 12、p = 8：ncv = 64，基 72 列，p = 1 时 29 列）。
  - **块步的正交化**（P5c，`BlockTuning::fast_ortho`，默认开）：没有 deflation 块时，每步先试单投影快路径
    `BlockOrthonormalizer::try_fast`——对 V 做行块化的块 CGS（与线程数无关；DGKS 判据要求时才做第二遍），
    再在块内做两次 Cholesky-QR（CholQR2，M·Q1 由 (M W1) R1⁻¹ 得到，不再作用一次 M）。它只接受列缩放后
    W1 的 2-范数条件数 ≤ `kMaxFastCond` = 64、没有列损失超过其份额、‖G2 − I‖_F ≤ 0.5 且
    |R(i,i)| > ratio·before(i) 的块；store_bv 模式总是拒绝。被拒绝的块、以及有 deflation 块（只在 1e-8 内
    正交归一，不满足快路径的前提）时的每一步，走两遍核 `run()`。每步块 M 作用：`try_fast` 2 或 3 次，
    `run()` 3 次（store_bv 2 次）。实测（`bench/blocktune_sweep.cpp`，16 线程测量槽，17 个 HEP/GHEP 测例，
    p = 2..8）：块运行的正交化时间降 28–76%，迭代时间降 3–24%，残差同量级、作用次数在逐次运行的波动内。
    `BlockTuning`（布局参数 extra_blocks、min_auto_ncv、keep_fraction 与 fast_ortho）与 `BlockRunStats`
    （各种步核的次数）只供测量与测试；16 线程的布局变体扫描是负结果（几何平均 −0.9%），默认布局不变。
  - driver 的用法（§3.6-5）：nearest 的位移反演运行用块宽规则（下），切片与单边运行仍用 P4 规则（nev ≥ 8
    时 p = 4，否则 1），证书收缩重跑宽度 max(q, min(4, 缺数上取偶))，q 为显式块宽 p、自动时 min(p, 4)。P3 的
    eigen 层测量（相同精度下几何平均 1.36–1.41×）由 driver 端到端测量取代（`results/p4_measurements.md` §3）。
- **块宽规则**（P5c，`src/eigen/block_rule.*`，eigen 内部、供 driver 用；纯函数，不读计时）：
  `recommended_block_size(ShiftInvertShape{n, nev, factor_entries, nnz_b, complex_scalars})`，
  factor_entries 为该运行位移处分解的因子条目（复数为实嵌入的），在 p ∈ `kRuleWidths` = {1, 2, 4, 8} 中
  选 步数比 × 每步代价 最小者：
  - 每步代价 `modelled_step_seconds`：一次 p 列回代 apply(p) = a + b·p（p = 1 时 1.1·(a + b)，MUMPS 的
    单右端回代走另一条 BLAS2 路径），a、b 由因子条目、n_eff（复数 2n）与小波前项 n_eff²/条目拟合；加一步
    块正交化（按自动 ncv 下平均基大小与 nnz(B) + n_eff）。常数在 16 线程测量槽标定（`bench/blocktune_sweep.cpp`，
    17 个测例），16 线程以外未专门测量。
  - 步数比 `relative_block_steps(p, nev)`：互不相同特征值时块步数相对 p = 1 作用次数之比，p = 2 / 4 / 8 在
    nev ≤ 5 时为 0.69 / 0.56 / 0.49、nev ≥ 22 时为 0.58 / 0.40 / 0.29，其间按 nev 线性插值（表取自
    nev = 1、2、3、5、7、10–12、21–26 的实测）。
  - 更宽的块须比每个更窄的候选便宜 `kRuleMargin` = 5% 才替换，宽于 `kRuleFloorWidth` 的（8 对 4）须便宜
    `kRuleWideMargin` = 10%（P5c 集成，见下）；块布局放不下（n 太小）的宽度不考虑；
    下限：nev ≥ `kRuleFloorMinNev` = 8 时至少 `kRuleFloorWidth` = 4（P4 的自动块宽）——重特征值在运行前
    看不见，而它们让更宽的块划算（低填充重特征值测例上 p = 4 比 p = 2 快 12–22%，低填充互异测例上 p = 2
    比 p = 4 快 0–8%）。
  - 结果（16 线程，标定时不带 BLAS hook）：61 个（测例, nev）点上所选宽度与最佳宽度的迭代时间之比几何平均
    1.035，没有一点慢于 p = 1；端到端（`bench/blocktune_fix_e2e.cpp`，12 个测例，5 次交错，155 次运行全部有效）
    相对 p = 1 快 1.03–2.45×，相对 P4 自动块宽 0.98–1.19×（ss-2cubes_sphere 选 p = 8 慢 1.6%）。driver 接入规则
    之后该程序的 `--p 0`（`--block-size 0`）就是规则本身，与 P4 块宽比较用 `--p 0,4`（k ≥ 8）或 `--p 0,1`（k < 8）；
    P5b 的 `bench/blocktune_e2e.cpp` 把最后一次分解（证书计数、重跑之后）的统计交给规则，已被它取代。
  - BLAS hook（§3.8，P5c 默认）的窄核使 p ≤ 4 的回代变便宜，模型因此高估了从 4 到 8 的收益：集成树中 8 对 4
    的实测每步时间比 1.27–1.33，模型 1.12–1.23。20 个定性 nearest 数据测例（CLI `--block-size 0` 对 4，gpu01
    测量槽 16 线程，5 次交错，200 次运行全部认证）按模型的 cost(8)/cost(4)：0.79–0.80 时 8 快 1.008–1.128×；
    0.88–0.89 时重特征值测例快（yee-pec-cube-32 1.084×、syn-mem27-46 1.047×），互异的慢（yee-bloch-1e4、
    yee-photonic-rods / -spheres-1e4 0.971–0.975×）；0.92–0.94 时都慢（yee-pec-box-1e5 0.982×、syn-lap3d-aniso-1e5
    0.991×、ss-2cubes_sphere 0.960×）。模型分不出重数，带 hook 重新标定会让 8 在互异谱上从不划算、也丢掉重特征值
    测例的 1.05–1.13×；所以只给 4 → 8 加宽余量到 10%：最后一组改选 4（比选 8 快：yee-pec-box-1e5 1.058×、
    syn-lap3d-aniso-1e5 1.008×、ss-2cubes_sphere 1.020×），其余保留 8；这三例与四个保留 8 的测例上相对 P4 块宽的几何平均 1.001 → 1.017。标定形状的选择在因子条目
    ±10% 内不变（选 8 者 cost(8)/cost(4) ≤ 0.898，改选 4 者 ≥ 0.907）。
  - 可复现性：p 只取决于输入。每次分析都给出同一排序时（AMD、METIS、自研嵌套剖分）因子条目、从而 p 逐次
    相同；SCOTCH（自适应排序在其得分最好时选它，§3.6-3b）的条目在三维数据测例上逐次运行相差 2.2–8.8%，处在
    两种宽度分界附近的形状可能在不同运行中得到不同 p（两者都有效、在容差内一致，同 SCOTCH 排序本身；标定
    形状的选择在条目 ±10% 内不变，测试 `LanczosBlockRuleFix.PinnedPicksSurviveTheOrderingSpread`）。
- `krylov_schur_general`：欧氏内积，实数用实 Schur 形（dgees/dtrsen），共轭对永不拆开：
  m ∈ {nev, nev+1}，共轭对相邻、正虚部在前，nconv 不拆对，schur_form 无被截断的 2×2 块，
  前 nconv 列 Schur 基张成（近似）不变子空间。**不支持 deflation**（非正规算子的不变子空间的
  正交补不是不变的；用法上以更大的 nev/ncv 重跑代替）；block_size < 1 抛 InvalidInput。
  - 结果向量：先形成 Schur 基 Q1 = V Q(:, 0:m)，释放 Krylov 基，再算 x = Q1 z（z 来自投影特征向量），
    峰值为 max(基 + Q1, Q1 + 向量)（wg-square-1e6：736 → 448 MB）；与旧的 V y 相差 ≤ 2e-15。
  - **块模式**（P4，`block_size = p > 1`，块 Arnoldi Krylov-Schur）：每步一次 p 列的 `op.apply`
    （多右端回代）；新块对基做块 CGS（`linalg::orthogonalize_block_cgs2`，DGKS 判据决定是否第二遍），
    块内两次 Cholesky-QR（`eigen_detail::block_cgs_cholqr2`）；某列崩溃、首个 Cholesky 因子的列缩放
    2-范数条件数 > 64 或第二个 Gram 阵偏离 I 时，恢复该块并交给两遍核 `BlockOrthonormalizer`
    （相关列换成确定性随机向量——块版的 Arnoldi 中断）。关系 Op V = V H + U C，C 为 p×m 耦合块：
    对的估计为 ‖C y‖₂，不变子空间检验用 ‖C Q1‖_F；选择、认证、实 Schur 形与输出与 p = 1 共用。
    快路径核比两遍核快 1.3–3.1×（p = 2..16；p = 64 为 1.0–1.3×），实测运行中从未回退
    （`results/p4_measurements.md` §4）。
  - 布局（`eigen_detail::block_general_layout`）：基数 = 单向量 ncv（自动或给定）上取整到 p 的倍数，
    且 ≥ s + 2p（s = nev + 2（实）/ nev + 1（复）上取整到 p 的倍数）；给定 ncv 时即为基数，自动时再加
    4p（受 n ≥ 2(ncv + p) 限制）。重启保留约 nev + (ncv − nev)/2 个 Schur 向量（p 的倍数、≥ s，共轭对
    处调整 1）。起始块：`initial` 的前 min(p, cols) 列 + 随机列。
  - 提前停止：在块边界做 Rayleigh–Ritz 探测（首轮从基数处起，之后从第一个新块起），nev 个全部认证
    即结束本轮；块运行因此恰好达到 tol，与 p = 1 须在相同残差下比较。
  - n < 2(ncv + p)（或 p 大到放不下一个块基）时用一轮覆盖整个空间的单向量循环精确求解（n 次作用，
    n×n 基：driver 把配置的 p 限制在 min(64, n/16)，直接调用 eigen 者自负）。
  - 重数 ≤ p 的特征值一轮内取全副本；超出 p 的只能靠舍入。内存：基 n(ncv + p) 标量，自动 ncv 更大
    （nev = 18、p = 8、实数：ncv = 72，p = 1 时 37）。
  - `krylov_schur_general_peak_columns(n, options)`：给定选项时的峰值 n-向量数（Krylov 基 + Schur
    基，与 Schur 基 + 结果向量取大；基为 p = 1 的 ncv + 1、块布局的 ncv + p 或整空间循环的 n + 1），
    按求解器同样规则校验选项；driver 的 GHIEP 用它做内存适配（§3.6-3）。
  - p = 1 与 a8ed8f4 逐位一致（Ritz 向量除外，见上；`test_krylov_schur_general_bitwise.cpp`：每种标量
    10 种配置，哈希表以平台指纹 + `/proc/cpuinfo` 型号为键——Xeon Silver 4314 与 Gold 6336Y 在配置
    10 的 OpenBLAS 乘积末位不同；未知平台只检查计数）。
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
    **driver 尚未调用二者**（P4 未接入；计划：JD + 单精度因子作最近 k 个的内存受限回退，LOBPCG 用于
    极低内存的最小/最大 k 个）。

### 3.5b verify（`include/hsolve/verify.hpp`，P4）
完备性检验的构件，用回代代替分解，供 driver 的策略 certificate = exact | randomized | auto 使用
（P5 起由 nearest 证书调用，`src/driver/certificate.*`，§3.6-8）。依赖 eigen（`lanczos_krylov_schur`）、
factor（InertiaCounter）、sparse、linalg；不打印；固定种子与线程数时确定。
- `exclusion_check(op, b_inner, X, σ, r, opt)`：B 正定（或 B = I），持有 K = A − σB 的分解，
  Op = K⁻¹B。问：除 X 张成的已找到特征对外，σ 的半径 r 内是否还有特征值（|θ| ≥ 1/r）？
  - 严格界：X 先 B-正交归一（Cholesky-QR，列均衡后的秩检验）；耦合 e = ‖Op X − X H11‖_B 由一次块
    作用算出（或调用方给出 `coupling_bound` 与 H11 的特征值）。Weyl：r 内多于 m_in 个特征值 ⇒ 收缩
    算子 H22 有 |θ| ≥ 1/r − e 的特征值（漏掉的重特征值副本亦然）。以收缩 = X、高斯起始块 S·G、
    s 个块步、无重启无提前停止的块 Lanczos 运行，**通过当且仅当** 补空间上所有 Ritz 值 |θ| < 1/r'
    （r' = `pass_radius` > r，须小于下一个合法特征值的距离）。Chebyshev（直接与平方两种，
    Kuczynski–Woźniakowski 式，适用于不定算子）+ 高斯起始分量的 Beta 分布给出
    P(通过且漏特征值) ≤ F(κ·t*(ε, s))^p，ε = 1 − (1/r')/((1/r − e)(1 − margin))，κ ≥ cond(S B S)；
    s 在运行**之前**由 `delta` 定出（`exclusion_block_steps`，倍增 + 二分），Krylov 空间覆盖整个
    补空间时 δ = 0。结果为 Passed / Detected（附新 Ritz 对）/ Inconclusive（ε ≤ 0、步数上限、κ 非有限、
    计划超资源预算）。
  - 资源预算：计划（`exclusion_run_size`）超过 `max_krylov_vectors`（512）或 `max_memory_bytes`
    （1 GiB）时不分配、不作用即返回 Inconclusive（窄谱隙下 s ~ log(Nκ/δ)/√ε 会规划 GB 级的基）。
  - κ 的严格性：`start_scaling_for(B)`：B = I 时 κ = 1；对角 B 取 S = B^{-1/2}、κ = 1（精确）；对角
    占优时 Gershgorin 上下界（严格）；否则只是估计（100 × Gershgorin 上界 / Lanczos 下界估计），
    结果标 `rigorous = false`、陈述写明「κ 为假设，筛查结果而非证书」。B = I 而给了 S 时检验自算
    κ = cond(S²)。`start_scaling_for(b, safety, estimate = false)`（P5）跳过一般 B 的 Lanczos 估计
    （n = 1e5 时约 0.1 s），返回 κ = +∞、`rigorous = false`——只要有证明的 κ 的调用方（driver）用它。
  - **一般 HPD B 的证明的 κ**（P8，`prove_condition_bound`，src/driver/norms.*；driver 的 certifier 用它）：
    W = diag(w)、wᵢ = fl(1/√bᵢᵢ)（证明所存的权，也就是起始缩放 S），则
    λ_min(WBW) ≥ lower = `prove_positive_definite` 的下界（§3.6-1(b)：对 WBW − cI 的不完全 Cholesky L，
    M = WBW − LLᴴ 的 Gershgorin 下界；M 由所存的浮点 L 与缩放元素算出，带 Higham 引理 3.1 的先验舍入界、2 倍余量，
    对所存的 B 与 w 严格成立，不依赖选主分解的增长因子）；λ_max(WBW) ≤ upper = maxᵢ Σⱼ |bᵢⱼ| wᵢ wⱼ（Gershgorin），
    每行算出的和乘 (1 + 2γ_{Nᵢ+4})：每项 fl(fl(|bᵢⱼ| wᵢ) wⱼ) 在精确乘积的 (1 + u)²（复数 |·| 另 2u）内，Nᵢ 个非负项
    的和在 γ_{Nᵢ−1} 内（Higham §3.1），因子 2 为余量。于是 cond(WBW) ≤ κ = upper/lower·(1 + 4u)（商的舍入向外），
    且 λ_min(B) ≥ b_min = lower/maxᵢ wᵢ²·(1 − 4u)（xᴴBx = yᴴ(WBW)y，y = W⁻¹x，‖y‖ ≥ ‖x‖/max wᵢ）。κ 只以 log κ
    进入随机检验的步数，Gershgorin 上界足够。证明失败（θ < 0.1、破缺、Gershgorin 不正、超预算）时不作结论。
    driver 只在随机检验或单精度计数（在 b_min 取最好可能值 min bᵢᵢ 时）可能比精确计数便宜、或模式为 randomized 时
    才算它；精确计数预测时间的 `kBoundsProofShare` = 0.25 是其 IC 尝试与 1 级填充模式的时间预算（各按预测时间
    预付），θ 的 Lanczos 估计与缩放副本 S 不计入，故整个证明可超出这一预算（16 线程、randomized、各 3 次：
    cav2d-rect-pert-1e5 0.016–0.023 s 对预算 0.016–0.021 s，ss-tem27623 0.026–0.048 s 对 0.024–0.032 s）；预算贴近
    尝试的预测时间时，用 1 级还是 0 级随实测分解时间而变（cav2d-rect-pert-1e5 一次 κ ≤ 9.1、两次 κ ≤ 35.8）。成功的
    证明在数据测例上 0.004–0.08 s。
    ss-qa8fk-qa8fm：lower 0.0358、upper 4.22、κ ≤ 118、b_min 1.26e-6（16 线程 0.084 s）；auto 在该例上仍选精确计数
    （随机检验 ε = 0.14、预测不比精确计数便宜；单精度规则的 η ≈ 254 远大于窗口宽 2），见 `results/p8_certcost.md` §6。
- 单精度计数规则：`single_count_validity` / `single_precision_certificate`：B = I 或对角 B > 0 时，单精度
  LDLᵀ 计数是 K + E 的精确惯性，‖E‖ ≤ c·g·u_s·‖K‖₁；把窗口两端各收缩 η = ‖E‖/min bᵢᵢ 后若所有已找到
  特征值（含误差棒）都在收缩窗口内且计数相等，则认证。唯一非严格输入是 c·g = 1e3
  （`SingleCountRule::backward_factor`；MUMPS 阈值选主无可用的先验增长界）。P5 的一次实测违反了它：
  yee-pec-box-1e5（16 线程，约 50 个进程中的 1 个）上单精度计数在离每个特征值都至少 2.96e3·u_s·‖K‖₁/b_min
  的端点数错 2 个——c·g 是没有先验上界的假设，调用方必须设防（driver：auto 的 10 倍余量、多于找到数时
  双精度复核，§3.6-8）。一般 B 只在调用方给出**证明的** λ_min(B) ≥ `SingleCountRule::b_min_proven` > 0（P8，
  `prove_condition_bound` 的 b_min）时适用：Weyl 论证在 B^{−1/2} 合同下同样成立，η = c·g·u_s·(‖A‖₁ + |s|·‖B‖₁)/b_min；
  否则不适用、不声称证书。`SingleCountRule::a_norm_1`（≥ 0 时）是调用方已知的
  ‖A‖₁，省去每个窗口一次 O(nnz(A)) 计算。
- `certificate_cost`：由已测得的分解时间、单次作用时间等预测 exact（主对象重分解对 / 计数器对）、
  single、randomized（p ∈ {1, 2, 4, 8} 中最快且计划在预算内者）的秒数并推荐；randomized 须比最好的
  精确方案快 `preference` = 1.2 倍才推荐。单精度路径的分析另计 `single_analysis_s`（< 0 时取
  counter_analysis_s）。driver（§3.6-8 的 auto）按计数器的状态填这两项：新建的 certifier 计数器推迟双精度实例
  （§3.4），单精度路径只付一次分析（P5b 时为两次）；计数器已做过单精度计数时为 0；`counter_analysis_s` 在其
  双精度实例分析过后为 0，之前仍计一次分析。
- 实测（2 线程，`results/p4_measurements.md` §5）：完整 X 全部通过、删掉一个副本或最远特征值全部
  检出；随机检验只在分解贵（三维填充）**且**谱隙宽时占优（syn-mem27-46 比认证的单精度计数快 3.3×，
  yee-pec-cube-32 1.3×），窄谱隙（yee-pec-box-1e5，ε = 0.15）与廉价分解（1e4 测例）上更慢；代价模型
  在 7 个测例上都推荐了实测最快的认证方法。

### 3.6 driver（`include/hsolve/solver.hpp`）
实现：`solver.cpp`（编排、输入检查、B 的正定性检验链）、`steps.cpp`（共用步骤：位移、计数、分析与预算、
自适应排序与其代价模型）、`definite.cpp`（HEP/GHEP：最近 k 个与窗口/谱切片）、`certificate.*`（nearest 的
快速完备性检验与 auto 选择，P5）、`norms.*`（范数估计与 B 正定性的严格证明）、`indefinite.cpp`（GHIEP）、
`internal.hpp`。

0. **计时契约**：`PhaseClock` 把 `hs::solve` 的每一时刻记入且只记入一个阶段（嵌套作用域取最内层），
   共五个互斥阶段：setup（输入检查、范数估计、分析之前的 B 廉价检验、问题类型判定、规划）、factor
   （所有符号分析——含自适应排序的 AMD 探测与其它候选的分析——B 正定性的严格证明，以及位移、B（正定性
   检测）、A（GHIEP 零主元计数）的数值分解）、iterate（Krylov 求解：回代、正交化、重启）、verify
   （Rayleigh–Ritz、残差、补救、选择、证书簿记）、certificate（HEP/GHEP 路径的全部 Sylvester 惯性计数
   `count_below`——证书端点、窗口端点、切点、簇检查——不论计数在主对象上重分解还是在 InertiaCounter 上
   做，含计数器自身的分析；P5 起还有 nearest 的其它完备性检验：随机排除检验、单精度计数与其双精度复核、
   auto 的选择。例外：主对象计数之后随机检验需要在 σ 处重分解时，该分解计入 factor）。因此
   `time_total_s == time_setup_s + time_factor_s + time_iterate_s + time_verify_s + time_certificate_s`，
   各项非负。GHIEP 没有惯性计数，time_certificate_s = 0；`inertia_check = off` 或 `certificate = off` 的
   nearest 亦为 0。
1. **诊断**：
   - **输入检查**（P5，`hermitian_input`）：非方阵、空矩阵抛 InvalidInput；任一存储值非有限（报出位置）或
     矩阵全零（A = 0：每个特征值都是 0；B = 0：没有有限特征值）抛 InvalidInput。厄密缺陷
     ‖M−Mᴴ‖_F/‖M‖_F 用缩放平方和计算（`io::hermitian_defect_scaled`，与 `hs::load_matrix` 同一规则，任意
     量级都准确），超 `hermitian_tol` 抛 NotHermitian；按值比较不是精确厄密（某个 m_ij ≠ conj(m_ji)）时
     取 `hermitian_part`。内存输入（Python）与文件输入因此在任何量级下处理相同（P4 时内存输入用未缩放的
     缺陷，元素 ≲ 1e-154 或 ≳ 1e154 时下溢 / 溢出）。规模、nnz、实/复（A 复 B 实时 `to_complex`）。
   - ‖A‖₂/‖B‖₂ 估计（~30 步 Lanczos 的最大 |Ritz 值|：它是 ‖·‖₂ 的**下界**，从而 η 是真实后向误差的
     **上界**，验收偏保守）、scale = ‖A‖₂/‖B‖₂。
   - `threads > 0` 时把调用线程的 OpenMP 线程数设为 threads，`hs::solve` 返回或抛出时恢复原值
     （`OmpThreadsScope`，P5：库调用不改变宿主进程的设置）。
   - **B 的正定性**（`problem = auto` 且给了 B；显式给定的类型不检验）：
     (a) **分析之前的廉价检验**（`test_b_definiteness`，O(nnz(B))，setup 阶段），依次：对角 B 看对角的符号；
         每行严格对角占优且对角为正 ⇒ HPD（Gershgorin）；任一对角元 ≤ 0（eᵢᴴBeᵢ ≤ 0）⇒ 非 HPD；B 的范数
         估计 Lanczos 有 Ritz 值 < −`kIndefiniteRitz`·‖B‖₂（1e-10）⇒ 非 HPD（不重正交的 Lanczos 的 Ritz 值
         只差 O(eps‖B‖₂) 地留在谱的凸包内）。Lee–Sun–Cendes 波导的 B 由最后一项立即判定（P4 时要分解 B 读
         惯性：wg-square-1e5 上 0.19 s，总时间的 10%）。判定了的类型决定分析前的内存规划与排序代价模型
         （§3.6-3、3b）；未定时内存按两种路径中较大者预留，排序按定性路径加一次 B 的分解定价。
     (b) **严格证明**（仍未定、且目标不是 smallest / largest——它们的谱估计本来就要分解 B——时；主分析之后，
         factor 阶段；`prove_positive_definite`，`norms.hpp`）：D = diag(B) > 0，S = D^{-1/2} B D^{-1/2}；
         在填充级 1（时间或内存不够时 0）上对 S − cI 做不完全 Cholesky 得 L，则 M = S − LLᴴ 的 Gershgorin 下界
         minᵢ(m_ii − Σ_{j≠i}|m_ij|) > 0 证明 λ_min(S) > 0（Weyl；LLᴴ 半正定，L 是什么都成立）。M 的元素由存储的
         浮点 L 与缩放后的元素算出，带严格的先验舍入界（Higham 引理 3.1，留 2 倍余量），所以正的下界是对所存
         B 的证明，不依赖选主分解的增长因子。位移 c = 0.9θ（θ：S 上 20 步 Lanczos 的最小 Ritz 值），某主元不为
         正（破缺）后再试 0.5θ、0.2θ；θ < 0.1 不尝试（这样的数据测例在每个填充级与位移上都失败，证明成功的
         都有 θ ≥ 0.14）；
         S 上 Lanczos 有 Ritz 值 < −1e-10·max|Ritz| 则证明 B 不定。**预算**：时间 `kBProofBudget` = 0.5 ×
         主分析估计下一次 B 分解的模型时间（`factorization_s`），每次尝试与填充级 1 模式的两个构造阶段都在
         运行前按预测时间扣除，余额不够即不做（S 上的 Lanczos 与 θ 过门槛后存储的 S 都是 O(nnz(B))，不计入
         预算）；内存 min(预算 − 当前占用, 分解上限)，在分配任何东西之前检查
         （S 只在能与最小的尝试一起放下时才存；填充级 1 模式先按 O(n) 的界、再按第一遍数出的精确条目数定价）。
         16 线程实测（`bench/anaint_bproof.cpp`）：失败的证明 0.001–0.09 s，最差比例是 ss-tem27623 的 0.054 s
         对 B 的分解 0.058 s；成功的 0.004–0.08 s，对 0.026–0.20 s。棱元铅笔（ss-tem*）上证明失败。失败的
         证明对 B 不作任何结论。
     (c) 仍未定：在主分析上分解 0·A + 1·B 读惯性（主对象为双精度非 BLR；否则 `compute_inertia`），B 为 HPD
         当且仅当没有负、零主元。
     事件 `b_inertia` 的 `how` 记录判定方式（diagonal、dominant、diagonal_entry、lanczos、
     incomplete_cholesky、lanczos_scaled、factorization、compute_inertia），证明的细节在事件 `b_proof`。
2. **策略**：
   - HEP/GHEP → 位移反演 + `lanczos_krylov_schur`（B 内积，块宽见 §3.6-5）：nearest 用单位移
     （§3.6-8）；interval / smallest / largest 用窗口与谱切片（§3.6-8b）。
   - GHIEP → 位移反演 + `krylov_schur_general`（块宽见 §3.6-5），**只支持 nearest**（其余目标需要
     惯性计数，抛 Unsupported）；零簇见 §3.6-9。
   - `method = lanczos`（免分解）抛 Unsupported；LOBPCG / JD 已在 eigen 层实现（§3.5），driver 尚未
     调用。
   - smallest / largest 的惯性搜索以谱两端的 Ritz 估计为种子：30 步 Lanczos（A；有 B 时为 B 内积下
     的 B⁻¹A，B 的分解复用主分析，对角 B 直接求逆）。
3. **内存预算**（P7：默认由机器决定）：`memory_budget_gb` 默认 `"auto"`（= 0，`kMemoryBudgetAuto`），
   取进程最小内存上限 L 的一部分：budget = max(0.5, min(0.85·L, L − 1)) GiB（`kAutoBudgetFraction`、
   `kAutoBudgetHeadroomGb`、`kAutoBudgetFloorGb`；`resolve_memory_budget`）。L 来自
   `detect_memory_limits`（util.hpp）：内存 cgroup（v2 `memory.max`、v1 `memory.limit_in_bytes` 与
   `memory.stat` 的 `hierarchical_memory_limit`，沿 /proc/self/cgroup 的路径取最小；容器内路径不存在
   时读挂载点根）、调用方声明的 `HSOLVE_MEMORY_LIMIT_GB`（GiB；`tools/run/guard.py` 把它设为本次运行
   的 RSS 上限：测量槽 32、构建道 12）、有限的 RLIMIT_AS / RLIMIT_DATA、MemTotal（不用 MemAvailable：
   决策不随别的进程此刻占了多少而变）；都读不到时 14 GiB（P7 前的固定默认）。显式数值（config / CLI /
   Python）仍优先；超过 L 时告警。报告 `memory{budget_gb, source, limit_gb, limit_source, measure,
   reserved_gb, limits[]}` 与事件 `memory_budget` 记录预算、来源与每个来源的值。系数的标定（峰值 RSS 对
   预算，最大测例）见 `results/p7_membudget.md`：计划内的量（规划时的占用、Krylov 预留、ICNTL(23) 上限下
   MUMPS 的估计）都在预算内，L 的其余部分容纳计划外的量（回代工作区、证书计数器的分析、碎片）。
   **度量**（P7 评审修正）：预算按 L 所限的量计（`MemoryBudget::measure`，`measure_of`）：cgroup、
   `HSOLVE_MEMORY_LIMIT_GB`、MemTotal 限常驻内存（RSS）；RLIMIT_AS 限地址空间（VmSize），RLIMIT_DATA 限
   VmData（Linux ≥ 4.7）。地址空间里有大块只预留不使用的部分：进程启动时（尚未计算）OpenBLAS 线程池的
   缓冲（gpu02，16 线程：VmSize 2.2 GB 而 RSS 6 MB；8 线程 1.1 GB），线程池启动后的 malloc arena 与线程栈
   （syn-mem27-80 载入后 VmSize 3.5 GB 而 RSS 0.18 GB），它们在规划前就已存在、不随问题增长；之后地址
   空间只随分析（+1.2 GB，RSS +0.8 GB）、MUMPS 的预留（cap 内）与 Krylov 基增长。所以每线程另留余量
   并无必要，而地址空间（或 VmData）上限下预算 = R +
   max(0.5, min(0.85·(L − R), L − R − 1))，R = 解析预算时该度量超出 RSS 的部分（`reserved_gb`）；预算在
   诊断（已跑过并行区，线程与 arena 已建）之后才解析。规划、分析后的重新规划与所有"预算 − 当前占用"
   （块宽、切片、B 的证明、随机检验）都用同一度量（`driver::memory_in_use` / `budget_left`）。配置的数值
   预算指常驻内存。
   P7 前固定 14 GB：1e6 的三维测例（syn-mem27-100，因子 15.9 GB）在 32 GB 机器上也走外存，274.6 s，
   而内存内 125.1 s。P7 实测（16 线程，3 次中位数）：32 GiB 上限下 syn-mem27-100 与 yee-pec-box-1e6
   改为内存内，比 14 GB 默认快 1.83× / 1.71×；16 GiB 下照旧外存、同速；1e5 测例决策与时间不变。
   分析前 factor_cap_MB = 0.95·(budget − 当前占用 − n·sizeof(T)·c_res)/1e6。
   预留列数 c_res（`krylov_reservation`，事件 `memory_plan` 记下预留的块宽、列数、MB 与 cap）按分析前已知的
   问题类型（§3.6-1(a)）：定性路径（HEP/GHEP）ncv_g + 6k_g，k_g = k（interval：min(n, 64)），
   ncv_g = max(ncv（自动时 max(2(k_g + 16) + 1, k_g + 32)），`lanczos_basis_columns`(n, 0, k_g, ncv, p_g))，
   p_g = `reserved_block_size` = min(block_size > 0 ? block_size : 4, 4)——每种目标都只预留 P4 块宽的基。
   更宽的块宽——显式的 block_size，或块宽规则在 nearest 运行上选的 8（§3.6-5，分解之后才知道）——不预留：
   分解优先，运行时只在因子留出的内存放得下其基时才用（见下）。P5c 集成时曾为自动块宽预留 p = 8 的基，cap
   因此少了 27 个 n-向量（k = 10..12）：syn-near-singular-complex-1e5（AMD，4 线程，因子内存内 142 MB、外存
   74 MB）在 0.3692 / 0.3706 GiB 的预算下，P4 的基旁 cap 为 143–145 MB、因子在内存内，p = 8 的基旁只有
   102 / 105 MB——因子转外存（`out_of_core = off` 时 OutOfMemoryBudget），而规则随后仍选了 4；故改回 P4 的预留。
   GHIEP 为 `ghiep_reserve_columns`（P5：第一次运行的 `krylov_schur_general_peak_columns`，块宽取
   `ghiep_block_size`，单精度/BLR 另加修正工作区 2p，再加后续
   运行分解期间保留的普通运行 Schur 基与并集精修基各 m 列，m = min(n, nev + 1（实）/ nev（复））；类型
   未定时取两者之大。cap ≤ 0 即抛 OutOfMemoryBudget。
   分析会常驻自己的结构（MUMPS 符号数据与输入副本：n = 5e5 的三维测例
   0.8 GB，1e6 时 0.9–1.2 GB，规划时看不到），P7 起分析之后 cap 减去进程此间增长的 0.95 倍再做决定
   （事件 `memory_replan`，`SparseFactorization::set_max_memory_mb`）。`analyze` 后估计 ≤ cap 才做内存内分解，cap 是
   MUMPS 工作区的上限（超限变成 INFO=−19 而不是被系统杀掉；实际给的 ICNTL(23) 见 §3.4 工作区预留）；
   否则 `out_of_core ≠ off` 且外存估计 ≤ cap 时
   改用外存；仍超则抛 OutOfMemoryBudget 并报告估计值（绝不盲目分解）——P8 起逐次不同的排序（SCOTCH，§3.4）
   先再分析（至多两次，告警）：ss-3dspectralwave2 在 14 GiB 预算下外存估计多为 10.6–11.9 GB，P5c 与 P8 的
   42 次分析中有 6 次 13.4–14.4 GB（其中 5 次超出约 13.49 GB 的上限；`results/p8_certcost.md` §2.3）。
   `out_of_core = auto` 时内存内
   分解若仍因内存失败（INFO −13/−19，或 cap 下工作区重试长不够——估计偏低，如延迟主元），改用外存
   重做并保持（`FactorOptions::ooc_on_memory_error`，`factor_stats.ooc_switches`）。外存运行仍需
   外存工作内存与证书的只计惯性计数（InertiaCounter 总在内存内，工作内存与外存估计相当：
   syn-mem27-100 7.2 GB 对 7.3 GB）都 ≤ cap。
   **非精确因子**（P7）：`factor_precision = single`（或 `blr = on`）的因子每次算子 apply 都经双精度
   修正，且受检查——修正未收敛（`DriverShiftInvert`，§3.4b）、非精确因子在某位移关掉零主元检测后
   仍奇异（单精度的检测阈值 1e-4‖K‖ 也会标出单精度分辨不了的近邻特征值：cav2d-rect-pert-1e5 有 16 个
   零主元，修正仍 3 步收敛；所以 `factorize_near` 对非精确因子不做扰动，而是关检测在 σ 处重分解，由
   每次 apply 的检查裁决），或结果不被接受（有对超 tol、证书失败）时，`driver::InexactFactorFailure`
   放弃这次尝试，从诊断之后以双精度、同一主元顺序（`set_pivot_order`，免第二次排序）重解（重解前
   `malloc_trim` 把放弃的尝试释放的堆还给系统，否则新的规划把它算作占用：n = 27000 时 57 MB 中的
   41 MB）；报告 `precision{requested, used, blr, decision, fallback, fallback_reason, attempt_s}`，
   放弃的尝试时间计入各阶段，auto 的 `decision` 与其告警保留。单精度因此只用于省内存（因子 0.51–0.67 倍），从不返回错误结果；它本身更慢（P6：端到端
   0.35–0.59 倍）。`factor_precision = auto`：分析（双精度）后若双精度内存内估计 > cap 而
   `out_of_core ≠ on`，以同一主元顺序分析单精度对象，其内存内估计 ≤ cap 才改用单精度
   （`choose_precision`），否则双精度外存。默认仍为 double；auto 的实测取舍见 `results/p7_membudget.md`。
   窗口目标（§3.6-8b）在计数之后另查：n×m 输出加每片至多 4 对余量须 ≤ 0.9·(budget − 当前占用)，
   否则 OutOfMemoryBudget；一片的工作集约 7·c·n·sizeof(T)（基 ~2c，Rayleigh–Ritz 的 X/AX/BX 与乘积
   临时量 4c，选中块 c）限制片大小 c；**无固定特征值个数上限**（P2 的 max(2000, n/2) 已废）。
   **块基的运行时适配**（P4）：HEP/GHEP 每次运行前 `fit_block_size` 把 p 减半（告警），直到
   `lanczos_basis_columns`（p = 1：d + ncv + 2；p > 1：d + 块布局 ncv + 2p，块布局放不下时 d + n_free + 2）
   加调用方保留的列（候选块 X、AX、BX 与返回向量：4·nev，重跑另加 3d）的字节 ≤ 0.9·(budget − 当前
   占用)。例外（P5c）：nearest 的自动块宽减到不低于预留的块宽（规则的 8 → 4）时不告警，只记事件
   `block_size_fit`（想要的与用的块宽、可用 MB）；再往下减、以及显式块宽与重跑的任何减半仍告警。
   GHIEP 同理：峰值列 `krylov_schur_general_peak_columns` 加修正工作区 2np（开修正时）
   ≤ 0.9·(budget − 当前占用)（当前占用含活的分解与此前运行的 Schur 基），每种减法只告警一次。更后面
   运行（探测、单边运行、重跑）的基不预留，由这里的减半适配。
   **GHIEP 后处理**（P4 前约 88 B·n·m，wg-square-1e6 上 +1.15 GB 的尖峰决定全程峰值）现在只保留各次
   运行的 Schur 基（Ritz 向量当即丢弃），投影铅笔按 8 列一块形成（不物化 n×m 的 AW、BW），候选按
   4 个一批精修并以精修基中的系数保存，结果向量只为选中的对、在各运行的基释放之后形成；每次精修前
   释放分解（§3.6-6）。wg-square-1e6（SCOTCH 固定，2 线程）峰值 RSS 2886 → 1766 MB（报告值；guard
   1.65 GiB），后处理 3.50 s → 1.29 s 且不再增长内存（`results/p4_measurements.md` §4）。
3b. **自适应排序**（`ordering = −1`，默认；P5c，`adaptive_analysis`）：先做 AMD 分析探测（极快，其估计区分
   二维型与三维型填充），再用代价模型把每个候选排序在本次运行**计划的工作**上定价；分析得分最好的候选、
   以其实际估计替换预测，如此往复，直到得分最好的是已分析的候选。
   - 候选（`candidate_orderings`，得分相近时按此顺序优先）：AMD（0）、SCOTCH（3）、自研并行嵌套剖分（10，
     §3.4）、METIS（5）、PORD（4）。SCOTCH 排在自研嵌套剖分之前：16 线程下它在高填充三维测例上的因子只需
     后者 0.80–0.95 倍 flops，而按填充比的形状拟合要到填充 ~150 以上才看得出；端到端把它换成自研嵌套剖分，
     yee-pec-box-1e5 慢 13%、ss-ge87h76 慢 17%。config 也可直接给 0、2..7（MUMPS ICNTL(7)）或 10，见
     `docs/FORMATS.md` §2。
   - 计划的工作（`plan_work`，由配置与分析前已知的问题类型推导，不计时；按 6a3d7c1 时数据测例的端到端
     运行标定）：HEP/GHEP nearest 分解 1 次（类型未定再加 B 的惯性 1 次）、证书计数 2 次（`inertia_check
     = off` 时 0）、列作用 40 + 4k（p > 1）或 2k + 12（p = 1），块宽按 P4 规则估计（分析之前还没有因子
     统计）；窗口目标（interval / smallest / largest）再加 1 次分解、计数 4 次（粗估，数据测例都是
     nearest）；GHIEP nearest 约 3.4k 列回代、1 次分解，σ 在零带内时 5 次分解、约 20k 列。只有候选之间的
     相对代价要紧：补救、重跑等对每个候选的价格影响相同。
   - 代价模型（`OrderingModel`，`ordering_model()`；常数在 gpu02 测量槽 16 线程、MUMPS 阶数 ≥ 5000 的 40 个
     数据测例上拟合，`bench/anaint_orderings.cpp`；`bench/anaint_fix_fit.cpp` 由原始输出复算每个常数并用
     driver 自己的代码重放选择：相对最佳排序的平均超额样本内 4.3%、留一测例 5.0%）。方括号内为预测/实测
     的范围：
     - 给定顺序的分析（MUMPS 带 PERM_IN：自研嵌套剖分交出排列、InertiaCounter 复用主顺序），也是每种分析
       的下限，[0.58, 1.54]；
     - 自研嵌套剖分：单线程功 W = w_v·顶点 + w_e·边 + w_fe·因子条目 [0.47, 1.38]；p 线程 W·(s + (1 − s)/p)，
       s = 0.383 为 16 线程实测的中位数（四分位 0.26、0.47；它不按单一串行比例的 Amdahl 定律扩展：4 线程时
       0.19）；复数铅笔至少一半的耦合为复数时排 n 顶点图（与 `compute_ordering` 的 Embedding::Auto 一致）；
     - MUMPS 的排序都是串行的：METIS、PORD、SCOTCH 为 ρ·W（输入图、各自的因子条目；ρ = 1.033、1.126、0.626；
       整个分析 [0.45, 1.0]，syn-* 的低填充块图除外，它们快 3–8 倍）；AMD 的排序为 a·阶数 + b·条目
       [0.25, 1.31]；
     - 各排序的 flops 与因子条目相对 AMD 之比是 AMD 填充比 f = 条目(AMD)/nnz(铅笔) 的函数：ln r 在 ln f 上
       分段线性（节点 f = 1、4、10、40、150、1000，节点外取常数；岭回归 0.05）。低填充（f < 5，块状或带状）
       嵌套剖分要 AMD 的 2–3 倍 flops，二维（f ~ 10）0.4–0.9 倍，三维（f ≥ 40）0.1–0.35 倍；flops
       [0.44, 1.86]、条目 [0.64, 1.36]；
     - 16 线程的数值阶段：一次分解 t_f = f_o·阶数 + f_e·输入条目 + f_fe·因子条目 + f_fl·flops [0.23, 1.79]，
       同一对象重分解 0.962 t_f，InertiaCounter 计数 0.886 t_f；单右端回代 t_s = s_o·阶数 + s_fe·因子条目
       [0.30, 1.44]，一次 p 列回代 (1 + 0.062(p − 1)) t_s；其它线程数乘 (16/线程)^γ（分解 γ = 0.46、回代
       0.22，由 16 / 4 线程的中位数比得到）；
     - 低填充（f ≤ 4）的 AMD 因子在 16 线程下的数值阶段远慢于其规模所示（没有分隔子的深窄消去树让 L0 层的
       线程空闲）：分解 × 1.86、回代 × 3.15，f 从 4 到 5 对数线性降到 1，按 ln(线程)/ln 16 缩放；
     - 证书计数按 driver 的做法计价（`counter_pays` 时计数器的给定顺序分析加计数，否则主对象重分解）。
   - 选择（`run_ordering_choice`，纯函数，可脱离 MUMPS 测试；`OrderingPricer` 定价）：已分析候选的得分是其
     计划的数值时间；未分析的为预测分析时间 + (1 − g)·预测数值时间，g 是拿到实际估计的期望收益（以 AMD 为
     锚时 0.05；以已分析的嵌套剖分为锚时 0.03——两种嵌套剖分之间的相对预测紧得多）；后面的候选须比前面的
     低 `kOrderingMargin` = 5% 才胜出。循环：分析得分最好的未分析候选，与保留的现任者按同一规则对决，败者的
     分析立即释放（同时至多两个分析常驻），再以新分析的候选为锚重新预测其余，直到得分最好的已分析。预测或
     估计超出内存内上限（§3.6-3）的候选排在所有放得下的之后、按大小排序（全都放不下的 out-of-core 运行因此
     不会把候选逐个分析一遍：ss-3dspectralwave2 上曾分析 4 个、50 s）。
   - SCOTCH 的排序逐次分析都不同（§3.4），因此**按预测采纳**（`taken_on_prediction`）：一旦得分最好就分析并
     采用（估计超出上限时除外），从不拿它的估计与别的候选比，也不作预测的锚。选择因此只是确定量（确定排序
     的估计、模型、线程数）的函数，相同的运行选相同的排序；墙钟时间只记日志（事件 `ordering`：每个候选的
     得分、是否分析及其分析耗时、计划的工作与选择）。
   - `deterministic = true` 沿用下面的 P4 规则（AMD 或固定种子的 METIS），保证逐位复现。
   - 已知偏差：ss-tem152078 上模型选了自研嵌套剖分（该例单线程排序实际是拟合 W 的 2.1 倍，syn-mem125-46
     1.9 倍，度、填充、分隔子大小都解释不了），其分析比 SCOTCH 慢一倍，总时间为 P4 的 1.15 倍。给嵌套剖分
     加上分析风险（未分析时加 0.5 倍预测分析时间）可修好这一例，但重放时 yee-bloch-1e4、yee-photonic-*、
     syn-lap3d-aniso-1e5、syn-mem27-46 与 ss-qa8fk-qa8fm 也转向 SCOTCH，平均超额 4.3% → 5.3%，未采用。模型只
     在 16 线程下标定（4 线程数据只用于线程指数与 W；它们是在测量槽有锁之前于槽核 28–31 上取的，复现时 4 线程
     计时也须持槽：`guard.py --exclusive --cores 16-19`，命令见 `bench/anaint_orderings.cpp` 头注释；
     `bench_anaint_decide` 的选择不依赖计时，在构建/测试道运行），低填充 AMD 的系数只来自 syn-* 测例。
   - **P4 规则**（`decide_ordering` / `nd_ordering`，现在只服务 `deterministic = true`，也是 bench 的对照）：
     AMD 探测后按确定的模型预估嵌套剖分在本次运行的分解与回代上能省的时间，超过其额外分析代价才改用嵌套
     剖分——SCOTCH，`deterministic = true` 时 METIS（5，固定种子）；该路径总是 deterministic，所以实际是
     AMD 或 METIS。模型：t_分解 ≈ 6e-7·order + flops/(12 GF/s·线程)，t_回代 ≈ 16 B·条目/12 GB/s；AMD 分析
     `model_analysis_s` = 3.4e-7 s·order + 2.5e-8 s·输入上三角条目 + 3.6e-10 s·AMD 因子条目（42 个数据测例
     的相对最小二乘拟合，order 为 MUMPS 输入阶数，复数为 2n）；嵌套剖分分析代价 = 7 × 该值
     （`kNdAnalysisOverAmd`）；AMD 填充比 > 20 视为三维型（ND 保留 0.35 flops、0.6 条目），否则（0.6、
     0.85）；预计分解次数 = 1 + 2（有证书）+ 1（有 B），回代次数 max(40, 3k)（interval 以 k = 32 计）。
     `model_analysis_s` / `model_factor_s` / `model_solve_s` 另供证书的 `counter_pays` 与 auto 选择（缺实测值时）
     使用。
4. **近奇异位移**（`factorize_near`）：`factorize` 返回 false（σ₀ 在数值上落在特征值上）时，先看 σ₀ 处的
   零主元数（事件 `singular_shift`；`near_null_eigenvalues`：实数即零主元数，复数实嵌入中每个特征值出现
   两次，取 ⌈零主元数/2⌉）：
   - 至多一个特征值的份：孤立特征值，**只扰动一次**：σ ← σ₀ + 1e-8·max(scale, |σ₀|)（孤立特征值此时已在
     检测阈值之外约 1e3 倍；`max_shift_retries = 0` 时不扰动），记入报告（shift_retries = 1，告警）。
   - 多于一个特征值的份（P5c），或扰动后的位移仍判奇异：σ₀ 在比检测阈值更密的簇内，与阈值同量级的扰动会
     改变「最近」的集合 → **在 σ₀ 处关闭零主元检测分解**（以后在此位移的重分解照做；shift_retries 报 2，
     `max_shift_retries = 0` 时 1）。先看零主元数，因为扰动后的位移是否被判奇异取决于主元顺序：
     syn-zero-cluster-real-1e5（σ₀ = 0 两侧 1e4 个间距 2e-9 的特征值，取 21 个）上 AMD 的因子判了奇异，
     METIS、SCOTCH 与自研嵌套剖分的没有；它们在 σ₀ + 1e-8 处找到的 21 对延伸到 2.85e-8，而离 0 那么近的有
     34 个（最近 21 个到 1.86e-8）：每个种子都要证书重跑，10–11 次分解而不是 6 次。
   - 关闭检测仍失败（精确零主元）：簇的情形最后试一次扰动位移；仍失败抛 Singular。
   「最近 k 个」的选择始终相对原始 σ₀。P2 的第二次 10× 扰动在该例上落到所需半径的 5 倍处（证书数出 166 个、
   找到 21 个），已废。有意放在或紧挨重特征值的位移（§3.6-8b 的部分簇与重根重跑，`on_eigenvalue`）不按簇
   判定、照常扰动，不告警、不计 shift_retries。
5. **迭代求解**：内层容差 = tol × inner_tol_factor；θ → λ = σ + 1/θ。nearest 恰好收敛 k 对
   （nev = k；完备性由证书负责，不靠多算余量）。主分解为单精度 / BLR 时算子带双精度迭代修正
   （≤ 10 步，rel_tol = 0.1·tol·inner_tol_factor；每次 apply 检查收敛，未收敛即改用双精度重解，§3.4b、§3.6-3）。
   **块 Krylov**（P4，`SolverConfig::block_size`，0 = 自动，1 = 单向量；报告 `block_size`）：
   - HEP/GHEP（`block_size_for`）：配置 > 0 时用配置值（显式值优先）。自动时：
     - nearest 的单位移运行（P5c）：在 σ 处的主分解（`factorize_near` 之后）上用块宽规则
       `recommended_block_size`（§3.5），形状取 {n, nev, 该分解的 `FactorStats::factor_entries`, nnz(B)（B = I
       时 0）, 标量类型}，p ∈ {1, 2, 4, 8}，nev ≥ 8 时至少 4，8 须比 4 有 10% 的模型收益。规则只读确定量（不做
       计时探测：选择、从而特征向量都不随机器负载变化；SCOTCH 排序下因子条目的逐次波动见 §3.5）。分析前的内存
       预留只按 P4 块宽（§3.6-3）：规则的 8 只在因子留出的内存放得下其基时使用，否则不告警地减到 4（再往下才
       告警）。依据（16 线程测量槽，§3.5）：标定时端到端相对 p = 1 快 1.03–2.45×，相对 P4 自动块宽
       0.98–1.19×；带 BLAS hook 的集成树中改选 4 的三例比选 8 快 1.008–1.058×，保留 8 的重特征值测例比 P4 快
       1.05–1.13×，yee-bloch-1e4 与 yee-photonic-*-1e4 选 8 比 P4 慢 2.5–3%。
     - 谱切片与单边运行（每片的 nev，含零带干扰时转交窗口的 nearest）：P4 规则，nev ≥ 8 时 p = 4，否则 1
       ——块宽规则只在 nearest 运行上标定。
     报告的 `block_size` 是第一次运行的块宽（经内存适配后）。证书收缩重跑缺 m 对时宽度为
     `rerun_block_size(cfg, p, m)` = max(q, min(4, m 上取偶))，q = p（显式块宽）或 min(p, 4)（自动块宽，P5c
     集成）：宽度 ≥ 重数时随机起始块保证一轮取全副本；上限 4（不设上限时
     syn-ghep-multiplicity-real-1e4 缺 9 个、p = 10 的重跑用 200 次作用而单向量只用 59 次），更多副本仍靠
     舍入与后续重跑。自动块宽进入时也先限到 4：重跑只求 m + 2 对，正是 P4 测得 8 慢于 4 的情形，而规则的 8
     是为主运行的 k 对选的（gpu01 测量槽 16 线程，5 次交错，`certificate_fault_drop = 1`，主运行用规则的 8；
     重跑阶段 = σ 处重分解 + 重跑迭代，宽度 8 → 4：ss-2cubes_sphere 1.301 → 1.197 s（作用 168 → 100）、
     yee-pec-cube-32 0.788 → 0.735 s、syn-mem27-46（auto）0.235 → 0.185 s、yee-bloch-1e4（auto）0.165 →
     0.131 s，范围不重叠；整次求解比集成树（重跑用 8、无 lead）快 1.05–1.33×）。P4 规则的依据（30 个
     HEP/GHEP 数据测例，4 线程，p = 1/4/8，`results/p4_measurements.md` §3）：三维 1e5 测例迭代阶段快
     1.4–2.5×，重特征值测例总时间快 1.4–2.2×（不再需要重跑）；k = 6、7（cav2d）块更慢（22 → 60 次作用；
     P5c 的块步快路径与规则使 p = 2 在这两例上也划算，§3.5），p = 8 只在重特征值主导时胜过 4。小而低填充的
     复数 1e4 测例最多慢 0.1 s；切片代价模型（§3.6-8b）仍按 p = 1 标定，窗口目标的块运行未做性能测量。
   - GHIEP（`ghiep_block_size`，indefinite.cpp）：配置 > 0 时取 min(p, nev, 64, n/16)，探测运行
     （nev ≤ 2）恒为 1；自动时 n ≥ 2e4 且 nev ≥ 4 的运行 p = 2，否则 1。依据（2 线程，
     `results/p4_measurements.md` §4）：p = 2 在九个测例上从未慢于 p = 1（1.06–1.58×），p = 4 最多慢
     1.5×（零簇旁的单边运行每块都带回 p 个零簇方向：wg-square-1e5 σ = 0 时整次求解 284/400/620/855 次
     作用）；n ≈ 1e4 时收益在噪声内。报告的 `block_size`（P5）是第一次运行（普通运行，或 σ 在零带内时
     的第一次单边运行；之前的探测运行按规则是单向量）经内存适配后的块宽，各次运行的块宽见事件日志
     `krylov_schur.block_size`。
   - 两条路径都在每次运行前按 §3.6-3 的剩余内存把 p 减半（告警；nearest 的自动块宽减到预留的 P4 块宽时
     不告警，§3.6-3）。
6. **精修**：HEP/GHEP 在原问题上做**秩揭示** Rayleigh–Ritz：B-Gram 矩阵 XᴴBX 做特征分解，丢弃小于
   最大特征值 1e-8 倍的方向（跨片去重时 1e-4），在剩余的 B-正交基上解投影铅笔。同一向量算两次、或
   补救把两列推到同一特征向量上，只留一份——「找到的对数」恒为不同 B-正交向量数，重复向量不可能
   凑满证书。GHIEP（`refine_union`）：各次运行的 Schur 基（与其 Ritz 向量同张成，实数 T 时为实向量）
   的并集做欧氏正交化（丢弃相关列，1e-8）得 W，按 8 列一块形成 G = WᴴAW、H = WᴴBW，经 LU 与复 Schur
   形解 H⁻¹G，一次精修全部候选（4 个一批，A·Re(e^{iφ}x) = Re(e^{iφ}Ax)，实投影不需额外乘积）；H 数值
   奇异（主元比 < 1e-12）时退回各次运行自己的 Krylov-Schur 对（x = Q1 z）。精修前由调用方显式释放
   分解（idle；之后的运行在别的位移上，或其回代远多于一次重分解；同位移的侧重跑或第二次
   recover_strip 因此多一次重分解，实测各例分解次数未变）。
7. **残差验证**：每对计算 ‖Ax − λBx‖₂、相对残差、后向误差 η（§4.5）。未达标者做残差修正迭代
   x ← x − K̃⁻¹(A − ρB)x（ρ 为 Rayleigh 商；对任意近似分解 K̃，其不动点都是 (A,B) 的精确特征对；
   每对至多 3 步，K(σ) 的分解按需重建），之后再做一次秩揭示 Rayleigh–Ritz；仍不达标则
   accepted=false。η > max(1e-6, tol) 的对视为幽灵，不计入找到数。
8. **完备性检验（nearest，HEP/GHEP；复数经实嵌入同样可用）**：方法由 `SolverConfig::certificate` 选（P5，
   `src/driver/certificate.*` 的 `NearestCertifier`，每次求解一个对象、跨轮保留起始缩放、计数器与失败过的方法；
   `inertia_check = off` 或 `certificate = off` 时不做，报告 note 为 "disabled by configuration"）。
   - 窗口（各方法相同）：r = max|λᵢ − σ|（选中的对），a ≤ σ − r、b ≥ σ + r，各自推到与更外侧最近的 Ritz
     提示值（ritz_all 的映射值）的中点；无提示时外推 max(1e-3·|端点 − σ|, 1e-8·scale)。
   - **exact**（默认）：两端的双精度 Sylvester 计数，count_inertia = ν(b) − ν(a) 与 count_found（窗口内不同
     B-正交向量数）相等即通过（计数规则与来源见下）。
   - **exact_single**（只由 auto 选）：同样两次计数，用 InertiaCounter 的单精度 LDLᵀ，只在 §3.5b 的扰动规则
     （B = I、对角 B，或 P8 起 λ_min(B) 有证明的一般 B）以 10 倍余量认证该窗口时——假定的后向误差 `kAutoSingleSafety` × 1e3 = 1e4·u_s·‖K‖₁
     （yee-pec-box-1e5 上实测的误数发生在 ≥ 2.96e3·u_s·‖K‖₁/b_min 处），每个已找到的特征值与所需区域都离端点
     η 以上。单精度计数多于找到数、或两侧与主分解的精确计数 ν(σ)（主分解为双精度、非 BLR、带零主元检测时）
     矛盾（ν(σ) − ν_s(a) 个应在 [a, σ)、ν_s(b) − ν(σ) 个在 [σ, b]）时，先在同一计数器上做双精度计数复核：
     两种精度一致的亏缺触发重跑；不一致说明单精度 LDLᵀ 的后向误差超出了假设，本次求解不再用单精度计数，
     由双精度计数下结论。单精度计数少于找到数、或某端判奇异时交给精确计数（重复 / 幽灵由它报告）。
   - **randomized**：在 σ 处已持有的分解上做随机排除检验（§3.5b `exclusion_check`）：以窗口内已收敛的已找到
     对（η ≤ 10·tol；选中的对取残差修正后的版本，加上其余计算出的对）作收缩，高斯起始块（每轮换种子），块宽由
     `certificate_cost` 选、步数在运行前由失败概率上界 `certificate_delta`（默认 1e-12）定。通过半径 r' = 窗口
     在较近一侧的半宽；声称半径 r_c 介于 r_edge = r + |σ − target.sigma|（σ 只在近奇异扰动后才不同于
     target.sigma）与 r' 之间：1/r_c = 1/r_edge − 0.01·(1/r_edge − 1/r')（`kClaimFraction`）。检验通过且 r_c 内
     找到的个数等于认证的个数时，以概率 ≥ 1 − δ，r_c 内恰有这么多特征值、都由已找到的对逼近，所以选中的对
     就是最近的 k 个；检出（补空间上有新的 Ritz 值）触发重跑；不能下结论（谱隙太窄、计划超出内存预算、
     收缩块被拒）时交给精确计数，以后各轮不再试。只在 κ(B) 有证明（B = I、对角 B、对角占优 B，P8 起还有
     `prove_condition_bound` 能证明的一般 HPD B，§3.5b）时运行：证明不了的一般 B 的 κ 只能估计，通过只是筛查结果——
     此时不运行，由精确计数出证书并告警，陈述写明原因；**筛查结果从不
     作为报告的结论**。最后一轮仍检出时由精确计数下最终结论（精确计数能通过的求解不会因随机检验失败）。
   - **auto**：每轮用 `certificate_cost`（§3.5b）以本次运行**实测**的主分解时间与 Krylov 运行一次回代调用的
     时间（按其块宽换算到单列）预测三种方法的秒数，选最便宜且能认证者：randomized 只在 κ 有证明、且比最好的
     精确方案快 1.2 倍（`preference`）时；exact_single 只在规则带 10 倍余量成立时；否则 exact。随机检验不能
     下结论、而单精度计数预测比精确计数便宜时，改用单精度计数。陈述前缀给出这次预测；事件 `certificate_choice`
     记下 `certificate_cost` 的全部输入（`bench/certint_model.cpp` 可用别的计时重算）。
   - **轮次**：每轮先做该模式的检验；结论「缺」（exact：计数 > 找到；exact_single：两种精度确认的亏缺；
     randomized：检出）→ 以已收敛对（η ≤ 10·tol）作 deflation、在 σ 处重分解后重跑（nev = 缺数 + 2，块宽
     `rerun_block_size`，§3.6-5），至多 2 轮，每轮之后同法再检；精确计数少于找到数 → 告警（重复/幽灵收敛）。
     重跑以 `lead_nev` = 缺数（§3.5，P5c 集成）运行：缺的那几个最近的对收敛即结束，多要的 2 对决定基、收敛了就
     保留，但不再拖住运行（下一轮照样检验窗口）。多要的对可能落在宽于内层容差的簇里：yee-pec-box-1e5（k = 10，
     删掉二重特征值 59.986 的一个副本）重跑的第三个对是 35475 重 curl-curl 零空间的向量，其估计停在
     3e-10·|θ|，高于内层容差 1e-11——无 lead 时重跑跑满 1000 次重启（约 200 s，整次求解本应约 5 s）；确定性复现
     （4 线程，max_restarts 60）宽度 4 / 8 的作用 1236 / 1976 → 40 / 72，16 线程测量槽中每次重跑一轮重启即止，
     整次求解 4.65–4.81 s。
     **选中的对少于 k**（补救合并了列、对未收敛、故障注入）时，任何检验都只覆盖已找到的对的窗口，第 k 个
     最近的特征值可能在窗口之外：即使检验通过也重跑（缺数至少为 k − 选中数），最后仍不足时陈述注明。
   - `certificate_fault_drop`（只用于测试与测量）：第一次 Krylov 运行之后、检验之前删掉这么多已找到的对（先删
     选中对中最大重特征值的副本、至少留一个，再删离 target.sigma 最远的选中对及其 Ritz 提示），检验必须发现
     它们缺失、重跑把它们找回。
   - 报告（`InertiaCertificate`，`docs/FORMATS.md` §3）：method（exact / exact_single / randomized / none）、
     rigorous、failure_probability、solves、seconds、statement；randomized 下结论时 count_inertia = −1、
     [lower, upper] = [σ − r_c, σ + r_c]、count_found 为 r_c 内找到的个数。
   - `count_below(s)` = ν₋(A − sB)：先带零主元检测计数；被判奇异时，s 若是谱隙点（证书端点）就在
     s 处关闭检测计数（稠密簇内任何移动都会越过特征值；向后稳定的 LDLᵀ 对离 s ≫ eps·‖K‖ 的特征值
     计数精确）；s 若是任意点（用户区间端点、插值切点、零带边界——可能恰在一个（重）特征值上）就在
     s ± step/10 处关闭检测计数（窗口用 step = 1e-10·max(scale, |s|)，即偏移 1e-11 相对）。仍奇异则
     向外移 step·4ⁱ（至多 8 次，计入 endpoint_moves）。全部计入 certificate 阶段（§3.6-0）。
   - **计数来源 `CountSource`**（P4）：主分解对象（在计数点重分解，之后持有 K(s)，Ctx::held——单边切片
     可直接复用为位移），或 InertiaCounter（首次计数时按主分析的主元顺序建立；只计惯性的分解，实测
     便宜 10–23%，不动主因子）。规则：
     - nearest 证书（每轮两次计数、主因子精确）：`counter_pays` 时用计数器——2·0.13·model_factor_s >
       model_analysis_s（计数器的分析约等于一次 AMD 分析；三维填充是，低填充的二维 / 复数嵌入否，与
       实测净节省的正负一致）；`inertia_counter = on / off` 覆盖。用计数器时**每次计数前释放主因子**
       （`MainFactors::Release`：证书之后只有重跑才要主因子，重跑在 σ 重分解一次——与主对象计数相同），
       两者从不同时驻留。单精度计数建立的 certifier 计数器（P5）之后也服务该求解的精确计数
       （`exact_counts`：不建第二个计数器；其双精度实例推迟到这里第一次双精度计数时才分析，§3.4；为复核单精度
       计数已数过的窗口不再计数）。
     - 主分解为单精度或 BLR：一律用计数器（双精度；取代 P3 的每次计数一次完整分析的
       `compute_inertia`），稠密簇内无检测计数因此也可用。
     - 窗口 / 切片：默认主对象（计数点兼作单边位移）；`inertia_counter = on` 或主因子不精确时用
       计数器，并只在主因子 INFOG(21) + 计数器估计超过分解上限时释放主因子（`MainFactors::KeepWithinCap`：
       切片之间要在别的位移上做 Krylov 运行）。是否释放在每次计数前重新判定。
     - 实测（4 线程，六个三维 1e5 测例，`results/p4_measurements.md` §3）：计数器比主对象计数省
       0.05–0.14 s 证书时间（5–13%，含其分析），峰值 RSS 比主因子常驻低 21–26%、比主对象计数低
       2–7%；证书仍有两次计数分解，占这些测例总时间的 30–40%（P4 时 hsolve 在真实三维 nearest 测例上慢于
       SLEPc 的主因；P5 的 exact_single 与 randomized 即为此而来）。
   - **计数失败不结束求解**（P8，`CountSource`、`NearestCertifier`、definite.cpp）：证书的某次分解失败——MUMPS
     内存不足（INFO −13、ICNTL(23) 上限下的 −19，`OutOfMemoryBudget`）、其它 MUMPS 错误（工作区重试用尽、−9 …，
     `FactorizationFailed`）或 `std::bad_alloc`——只淘汰该计数来源，不淘汰求解（Internal、InvalidInput、Singular
     仍照常抛出）：
     - `CountSource::count`：首选来源失败（事件 `count_source`、告警）后，本次求解不再用它，同一计数改在另一来源上做：
       计数器失败 → 主对象重分解（主分解精确时）；主对象失败 → 计数器。计数器在**每次计数前**先查其分析估计
       INFOG(16)：超过计数余量 `count_allowance_mb` = min(0.95·(memory_budget − 当前占用), 分解上限)（占用按预算的度量，§3.6-3） 时先释放常驻的
       主因子再查，仍超就不尝试（在 ICNTL(23) 下它要么 INFO −19，要么——MUMPS 刚好在上限下时——让进程超出预算：
       ss-3dspectralwave2 在 14 GiB 下计数器估计 14.5 GB 对上限 13.5 GB 失败，13.6 GB 时运行成功但峰值 RSS 15.1 GiB）。
     - 主对象**替补**首选的计数器时（内存已紧）同样先查余量（事件 `main_count_check`）：先释放其因子，其上次分解实际
       用的工作内存 INFOG(21)（尚未分解过时用分析估计）超过计数余量就不尝试——主对象自己的 ICNTL(23) 上限定于分析时、
       在 Krylov 基与已得对之前，挡不住这时的计数超出预算。yee-pec-cube-32 类比（METIS，0.394 GiB，8 线程）：P8 初版
       在主对象上计数并认证，但峰值 0.54 GiB、超预算 38%；现在 231 MB 对余量约 160 MB，不尝试，证书「未完成」、峰值
       0.31 GiB。INFOG(21) 是预测而非上界：该小例一次计数使 RSS 增加约 330 MB（比 INFOG(21) 多 43%，接近 MUMPS 所分配的
       INFOG(18) 316 MB），故恰在阈值附近仍可能超出：0.47 GiB 预算下第一次计数通过检查（231 对 234 MB）、峰值
       0.53 GiB（超约 13%），其后 RSS 不回落（JOB=−4 之后仍高约 160–175 MB），第二次被拒，证书未完成。
       ss-3dspectralwave2（14 GiB，16 线程）：故障位 4 强制计数器放不下时与默认配置下计数器估计超出余量时，两次主对象
       计数的 INFOG(21) 9.5 / 10.0 GB 都在余量 11.9–12.0 GB 内（计数前 RSS 2.43–2.47 GB），都认证，峰值
       12.07 / 12.77 GiB（`results/p8_certcost.md` §2.3）；INFOG(18) 在该例为 12.3–12.6 GB，若以它为准反会拒绝能放下的
       计数。主对象为首选来源时（窗口目标；nearest 在 `counter_pays` 为假时）照旧只受其 ICNTL(23) 约束，与求解自己的
       分解相同（同样可能因分析之后增长的 RSS 而超出预算，`results/p8_certcost.md` §7）。
     - certifier 自己的计数器（单精度计数及其双精度复核、`exact_counts`）同样先查余量（单精度计数用
       `InertiaCounter::estimate(Single)`，§3.4）；失败后本次求解不再用它（`counter_failure()`），由精确计数下结论；
       失败属内存时 definite.cpp 让 `CountSource` 也不再建同一主元顺序上的计数器（`avoid_counter`），直接在主对象上数；
       属其它 MUMPS 错误时 `CountSource` 仍先试自己的计数器。
     - 随机检验需要的 σ 处重分解、或其回代失败（hs::Error 中的内存 / MUMPS 错误，或 `std::bad_alloc`）：不能下结论，
       交给精确计数（同 P5 的不能下结论）。
     - 两个来源都不能计数（`CountFailure`）：错误码在每个失败的来源都因内存（OutOfMemoryBudget、`std::bad_alloc`、
       估计超余量）时为 `out_of_memory_budget`，否则为 `factorization_failed`——与来源的尝试顺序无关。nearest 返回
       已收敛的对，报告证书「已做但未完成」：performed = true、passed = false、rigorous = false、failure_probability = 1、
       count_inertia = −1、count_found = 窗口内找到数、note 以 "not completed" 开头、statement 写明原因与「对未经认证」，
       告警，success = false（CLI 退出码 1，结果照写）。收缩重跑在 σ 处重分解失败时同样返回已找到的对（事件
       `rerun_failed`、告警、statement 注明）。窗口目标在数窗口端点与切点时（尚无片解出）照旧抛出该错误；从第一片起，
       一片所需的计数（单边锚点的谱隙点、部分簇的两次验证计数）不可能时只放弃该片（事件 `slice_not_completed`、告警、
       该片 `slices[i].passed = false`、count_found = 0），其余片照常求解，返回已完成各片的对：证书 method = exact、
       passed = false、rigorous = false、failure_probability = 1、count_inertia = 窗口计数、count_found = 找到数，
       note 以 "not completed" 开头并列出未完成的片，statement 说明，success = false。
     - 故障注入 `certificate_fault_count`（只用于测试与测量，`docs/FORMATS.md` §2）：位 1 = 每个 InertiaCounter 的
       ICNTL(23) 设为 1 MB（计数在 MUMPS 内真实地 INFO −19）；位 2 = 主对象计数失败；位 4 = 计数器的计数余量为 0
       （预判放不下）；位 8 = σ 处的重分解（收缩重跑前、随机检验前）失败。失败种类：默认 OutOfMemoryBudget，位 16 =
       其它 MUMPS 错误（FactorizationFailed；位 1 改为每次计数前抛出），位 32 = `std::bad_alloc`（16 与 32 互斥）。
       位 64 = 位 1、2 只从窗口目标的第一片起生效（nearest 自身的证书不受影响）。`tests/cpp/test_certcost_faults.cpp`
       对实数与复数覆盖每条路径（两者都失败、各失败种类与错误码、窗口目标在片中失败、certifier 计数器的内存与非内存
       失败、主对象替补的余量检查）。
     - 实测：yee-pec-cube-32（METIS）在 0.394 GiB 预算下复现了 ss-3dspectralwave2 的失败方式（主因子外存、计数器估计
       300 MB 超出余量；P5c 代码 rc = 2 「INFO(1)=−19」）；P8 初版改在主对象上计数并认证但超出预算，修正后两源都不尝试、
       证书未完成、不超预算（见上）。ss-3dspectralwave2 在默认 14 GiB 下（`results/p8_certcost.md` §2.3）：计数器估计
       10.5–15.1 GB，21 次运行中 9 次超出余量、改在外存主对象上计数（慢约 10%；这 9 次的 INFOG(21) 9.4–10.1 GB 低于
       当时的余量 11.2–11.4 GB，修正后的检查同样放行），21/21 全部认证、峰值 RSS ≤ 12.9 GiB（P5c：15.1 GB 估计时
       INFO −19 丢失结果，13.6 GB 时峰值 15.1 GiB）；主分析的外存估计超上限另由 §3.6-3 的再分析处理。
   - 窗口目标（interval / smallest / largest）总是精确计数——它们的计数点就是切点与位移，`certificate =
     randomized` / `off` 被忽略并告警；零带干扰时转交窗口的 nearest 同样精确计数。报告的 method 为 exact。
   - GHIEP 无惯性证书（报告中注明，method 为 none）；P0 设计中「换种子、更大 nev 重跑」的延迟重启检查未实现。
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
     可能算到另一侧），交给去重。缺对时以已得对作 deflation 重跑（不用 `lead_nev`：片内缺的对不一定是排名
     最前的），只要还有新对（至多 缺数 + 4 轮、上限 32 轮、连续 2 轮无新对即停）；已得对中有重特征值时，
     重跑位移放在离它 1% 谱隙处（重根主导算子约 100 倍，舍入补出其余副本；正放在重根上会把数值重簇拆成
     逐个收敛的 θ）。
   - P4：计数默认在主对象上（§3.6-8）；每片的块宽为 `block_size_for`(该片 nev)（≥ 8 对时 p = 4），
     片大小的代价模型未按块运行重新标定，窗口目标的性能未重测。
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
     在其实部的 1/4 处做一次单边恢复运行（`recover_strip`），未决时告警且 success=false。全部运行
     一次性在投影铅笔上精修（§3.6-6），选离 σ 最近的 k 个带外特征值；某侧收敛范围未覆盖最终半径时
     加大 nev 重跑该侧。各运行的块宽见 §3.6-5（探测恒为单向量）。零簇模式在稠密消逝谱上无论 p 都慢
     （wg-square-1e5 σ = 0：p = 1 时 294 次作用；内缘探测不收敛，各约 50 次作用），是 P3 的位移放置，
     有待改进。
10. **GHIEP 实/复分类**：|Im λ| > complex_tol·max(|λ|, scale) **且** 实投影对未通过后向误差测试而
    复对通过，才算复特征值；复特征对单独输出（`complex_eigenvalues`），实特征对不受影响。
11. **输出**：见 `docs/FORMATS.md §3`（含 `slices`、`zero_cluster_dim`、`zero_modes_filtered`、
    五个互斥阶段计时，P4 起另有 `solver`（"hsolve"）、`version`（`hs::version()`）、`block_size`、
    `time_certificate_s`，P5 起 `certificate` 另有 `method`、`rigorous`、`failure_probability`、`solves`、
    `seconds`、`statement`；Python 接口的报告同源）。`factor_stats` 只统计主分解对象，InertiaCounter 的
    分析与计数只体现在 `time_certificate_s` 与事件日志（`inertia_counter`、`main_factors_released`；P5：
    `certificate_choice`、`randomized_check`、`single_counts`、`certifier_double_counts`）。P5c 另有内存与块宽的
    事件 `memory_plan`（分析前预留的块宽、列数、MB 与分解上限，§3.6-3）与 `block_size_fit`（自动块宽不告警的
    减半）；每次 Krylov 运行的事件 `lanczos` 记下块宽与 `lead`（§3.6-8）。

### 3.7 util（`include/hsolve/log.hpp`、`include/hsolve/util.hpp`）
- `Logger`：JSON lines 写文件 + 人类可读 stderr；事件名小写下划线，字段扁平。
- `Timer`、`peak_rss_bytes()`、`current_rss_bytes()`、`max_threads()`、`build_info()`。

### 3.8 blas（`src/blas/`，P5「asmkern」；不属于 libhsolve；实测与用法见 `docs/BLAS.md`）
- 作用：替换 `dgemm_`、`dtrsm_`、`dgemv_`（另包装 MUMPS 用到的 level-1 例程——只计时、原样转发——与
  `dmumps_c` / `smumps_c`——只为剖析器标出分析、分解、计数、回代阶段），按形状把选定的调用类改道到别的现有核；
  其余调用（以及配置为 off 时的全部调用）原样转发给查找顺序中位于 hook 之后的真实 BLAS（`next_definition`：
  dlsym(RTLD_NEXT)；Python 下 MUMPS 与 BLAS 由 dlopen(RTLD_LOCAL) 打开，则在已加载对象中查找）。标量集：只有
  double——MUMPS 的实数与复数（实嵌入）分解、回代都是双精度实数运算；单精度（smumps）不改道。
- 路由（环境变量 `HS_BLAS_ROUTE`，逗号分隔，每进程读一次；未知记号使进程带消息终止，拼错的路由不会悄悄运行）：
  `gemv1`（N = 1 的 dgemm → 真实 dgemv）、`strsm=<m>`（小三角求解核）、`trsv=<n>` / `trsv1`（窄 dtrsm → 逐列
  真实 dtrsv）、`narrow`（2 ≤ N ≤ maxn 的 NN / TN dgemm → 自带窄核：A 只流一遍、不打包，AVX-512F 或 AVX2 + FMA）、
  `par`（并行区外、A 至少 2 × minkb 的调用按固定块分给至多 threads 个线程；行太少的 AᵀB 按 K 切分，部分积按
  块序相加）、`all` = `gemv1,strsm=512,narrow,par`，另有参数 `maxn`、`rows`、`minkb`、`threads` 与指令集上限
  `avx2`、`portable`。未设该变量时用构建默认值 `HSOLVE_BLAS_ROUTE_DEFAULT`（CMake 缓存变量，默认 "all"，""
  = off；也作用于 `libhsolve_blashook.so`）；`HS_BLAS_ROUTE=off` 在运行时关闭全部路由（纯转发）。
- 指令集：运行时由 CPUID 选 AVX-512F、否则 AVX2 + FMA、否则可移植循环（`hsolve_blas` 不用 `-march=native`
  编译，SIMD 核自带 target 属性；非 x86 只编译可移植循环，`narrow` 至少要 AVX2）。`HS_BLAS_NO_AVX512`、
  `HS_BLAS_NO_AVX2`（设为 "" 与 "0" 以外的值）强制回退，供测试与对比。`HS_BLAS_PROFILE=<文件>`（`%p` 换成
  进程号）在任何配置下开启逐调用剖析器：按（例程、操作数标志、尺寸档、MUMPS 阶段、是否在并行区内、路由）
  累计次数、秒数、flops 与最少搬运字节，进程退出时写成一个 JSON（`bench_asmkern_report` 排版）。
- 构建：总是构建静态库 `hsolve_blas`（路由、核、剖析器；`-ffp-contract=off`，使 NN 的两条代码路径舍入一致）与
  `libhsolve_blashook.so`（LD_PRELOAD 用，任何可执行文件，含 python）。CMake 选项 `HSOLVE_BLAS_HOOK`（默认 ON）
  经函数 `hsolve_link_blas_hook` 把入口点链入 hsolve 可执行文件：`--no-as-needed` 让 MUMPS 库留在 DT_NEEDED
  （可执行文件自己的 dmumps_c 满足了 libhsolve 的引用，conda-forge 的 `--as-needed` 会丢掉 MUMPS 库，包装便找
  不到真实的 dmumps_c），动态列表导出入口点，使 MUMPS 的 BLAS 调用绑定到它们。链入与预加载二者择一（否则调用
  两次经过 hook）。libhsolve、单元测试（`test_asmkern_fix_exehook` 除外，它检验这种链接）与 Python 模块都不带
  hook；Python 下用预加载（`docs/PYTHON.md` §1）。已有构建目录保留缓存的旧值，用
  `-UHSOLVE_BLAS_HOOK -UHSOLVE_BLAS_ROUTE_DEFAULT` 重新配置才取新默认值。
- 结果与线程：路由只改变调用内的运算顺序（与真实例程差在舍入），不改变语义；固定线程数时每条路由都确定（每个
  输出元素由一个线程按固定顺序计算，切分只取决于形状与线程数），不切 K 的窄核对任何线程数结果相同。OpenMP
  并行区内（MUMPS 的 L0 子树）每条路由都在调用线程内串行，不与 MUMPS 自己的线程超订；并行路由的固定块轮转分给
  运行时实际组成的线程组，线程组较小（OMP_THREAD_LIMIT、OMP_DYNAMIC 等）时仍算全部块。任意多线程可并发调用。
- 默认（route all 链入 hsolve）的依据：16 线程测量槽、六个对比配置、5 次交错，每次运行都有效且精度相同，
  端到端 1.05–1.16×；收益全在回代（MUMPS 回代时间降 18–30%，数值分解不变）。换用其它 BLAS（自编 OpenBLAS、
  BLIS、LIBXSMM）无益（`docs/BLAS.md` §3、§5、§6）。
- 许可：窄核改编自 OpenBLAS 0.3.34 的 SkylakeX 核（BSD-3-Clause），版权与许可声明见
  `src/blas/simd_kernels.hpp`、`docs/BLAS.md` §7 与根目录 `THIRD_PARTY_NOTICES`。

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
Krylov 求解器：Ritz 估计 ≤ tol_inner·|θ|（θ = 0 时用最大 |Ritz 值|）；两种 Krylov-Schur 的块模式都在
块边界也检查（§3.5），因此块运行恰好达到 tol（driver 的验收按真实后向误差 η ≤ tol，§4.5）。
LOBPCG / JD 直接以后向误差 η ≤ tol 判收敛（§3.5、§4.5）。

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
- **禁止**：稀疏矩阵稠密化（唯一例外：测试/评测代码中 n ≤ 4000 的参考解）；全局可变状态（例外：
  串行化 MUMPS 调用的进程级锁，§3.4；BLAS hook（§3.8，不属于 libhsolve）的剖析器——只在 `HS_BLAS_PROFILE`
  开启时记录——的每线程表与进程级 MUMPS 阶段标记；稀疏核分派的 CPU 检测与 `HS_KERNELS_NO_AVX512`、BLAS hook
  解析出的真实例程与路由配置都在首次使用时写一次、之后只读，不算可变状态）；无种子随机数；库代码中
  `std::cout`/`printf`（BLAS hook 只在无法继续时——找不到真实例程、未知路由记号——向 stderr 写一条消息并终止）。
- 测试：`tests/cpp/test_<module>*.cpp`（GoogleTest），**套件名以模块名开头**（`Sparse*`、`Io*`、
  `Linalg*`、`Factor*`、`Ops*`、`Lanczos*`、`KrylovSchurGeneral*`、`Lobpcg*`、`JacobiDavidson*`、
  `Verify*`、`Driver*`；P3 新增如 `LinalgDenseOpsBlockCgs`、`FactorInertiaCounter`、`LanczosBlockTyped`、
  `LanczosBitwiseTyped`、`LobpcgTyped`、`JacobiDavidsonTyped`、`DriverSlicing`、`DriverZeroCluster`；P4 新增
  如 `SparseKernelPaths`、`OpsKernelPaths`、`OpsFactorSolve`、`FactorOrdering*`、`FactorBlockAnalysis`、
  `KrylovSchurGeneralBitwise`、`VerifyExclusion`、`VerifyBound`、`DriverBlocks`、`DriverCounter`、
  `DriverGhiepBlock`；P5 新增如 `DriverCertificate`、`DriverHpdProof`、`DriverBDefiniteness`、
  `DriverAdaptiveOrdering`、`DriverOrderingModel`、`LanczosBlockRule`、`VerifyStartScaling`，blas 模块为
  `BlasHook*`；P5c 修复轮的 `AnaintFix*`、`AsmkernFix*`、`CertintFix*` 与 P5c 集成的 `BlocktuneDriver`
  （driver）、`CounterDeferred`（factor）、`P5cIntFix*`（eigen 与 driver）以子任务代号开头，是本规则的
  例外）。逐位哈希测试的参考表以平台指纹（+ CPU 型号）为键，未知平台只查计数；ctest 名为
  `<可执行文件>.<套件>.<用例>`，`tools/dev/test.sh release test_factor` 只跑该文件，匹配不到测试即
  报错。共享夹具在 `tests/cpp/support/`（`hsolve_test_support`），只追加不改签名。
  数值测试尽量用有解析解的问题。Python 工具与 Python 接口的测试在 `tests/py/`（pytest；
  `test_python_binding.py` 默认导入 `build/release/python`，`HSOLVE_PYTHON_DIR` 可改）。
- 测量程序：`bench/<name>.cpp`（目标 `bench_<name>`，见 `bench/README.md`），提交信息与文档中引用的
  数字都应能由其中某个程序经 guard.py 复现。
- 每个模块的测试须调用其头文件声明的每个模板的每个标量实例（缺实例化在模块内就暴露）。
- 新增依赖必须是开源许可，并登记到 `docs/DEPENDENCIES.md`。

## 6. 资源与实验规范
**计算位置**（P4 收尾起）：本机（开发服务器，共享，2×Xeon Gold 6336Y，48 物理核，125 GB）不再做
编译、测试与实验，只做编辑、git 与合并；全部计算在集群节点 gpu01 / gpu02 上进行，c001–c005（与
gpu 节点同型，无 GPU，与其他用户共用）只在空闲时使用：
- 节点：2×Xeon Silver 4314（2×16 核，AVX-512），251 GB，CentOS 7（glibc 2.17），无外网，home 目录
  各节点共享。`tools/dev/deploy_cluster.sh HOST` 由本机 conda 包缓存离线重建两个环境
  （`docs/DEPENDENCIES.md`）、建裸仓库（git remote `cluster`）与克隆、拷贝测例数据。
- `tools/dev/cluster_run.sh HOST [--wip | --allow-dirty] -- CMD`：把当前 worktree 的 HEAD 推到
  `wt/<name>`，在节点的 `$HOME/os2026-wt/<name>` 检出该提交，经 `env/run.sh` 运行 CMD；`--wip` 用
  `git stash create` 的临时提交传送未提交的已跟踪改动（不动工作区、索引与 stash 列表）；调用方的
  `HS_CORES`、`HS_JOBS`、`HS_MEM_GB`、`HS_MIN_AVAIL_GB`、`HS_LOCK_WAIT_S` 被转发（调度上限不转发）。
  同一 `<name>` 不得同时在两个节点上构建（home 共享）。
- **节点调度**（`env/hosts/<host>.sh`，`env/activate.sh` 按短主机名加载；由 guard 强制执行）。单次运行的
  规格与评测机一致：16 核、32 GiB。同一节点上同时跑多个这样的运行会互相拖慢、计时失真，也会把机器
  占满，因此每个节点分两条道：
  - **测量槽**：NUMA node 1 的 16 个物理核 `16-31`（`HS_MEASURE_CORES`），同一时刻只允许一个运行。
    凡绑到这些核的运行都必须 `guard.py --exclusive`（节点本地 flock）。不给 `--cores`/`--mem-gb` 时
    默认取整个槽与 32 GiB（`HS_MEM_CAP_GB`）；须给 `--timeout`，且不超过 `HS_SLOT_MAX_S=2700` 秒，
    防止一个运行长期占槽，长扫描要拆成多次。等锁超过 `--lock-wait`（默认 `HS_LOCK_WAIT_S=300` 秒）
    返回 75 并打印持有者（目录、命令、开始时间）；`guard.py --status` 查看槽是否空闲。少于 16 线程的
    缩放实验也在槽内做（如 `--cores 16-19`），以独占整个 socket。
  - **构建/测试道**：NUMA node 0 的 8 核 `8-15`（默认 `HS_CORES`），`HS_JOBS=4`，单次 RSS ≤ 12 GiB
    （`HS_MEM_GB`、`HS_MAX_SHARED_MEM_GB`）。用于编译、单元测试与冒烟运行，不用于计时。
  - 任何运行 RSS 都不得超过 32 GiB。每节点最多两个调用方时，我们最多占 24/32 物理核、约 56 GiB，
    节点其余资源留给其它作业。
  - 选节点：`tools/dev/cluster_nodes.sh` 列出各节点的槽状态、槽所在 socket（含超线程）与整机的忙碌
    逻辑 CPU 数和可用内存；`--pick` 给出一个空闲节点（槽空闲、该 socket 无忙碌 CPU、可用内存 ≥ 64 GiB）。
    guard 的忙碌检查也覆盖被绑核的超线程兄弟。c001–c005 上有其他用户作业时不在该节点计时。
- 两种 CPU 上 OpenBLAS 的乘积末位可能不同（§5 逐位测试以 CPU 型号为键）。
- **BLAS hook 与测量**（P5，§3.8）：默认构建的 `build/<preset>/hsolve` 带 hook（`HSOLVE_BLAS_HOOK=ON`，
  route all），计时的就是它；单元测试与 Python 模块不带。要测纯 OpenBLAS 的 hsolve，运行时设
  `HS_BLAS_ROUTE=off`，或另建 `-DHSOLVE_BLAS_HOOK=OFF` 的构建目录；不带 hook 的可执行文件可预加载
  `build/release/libhsolve_blashook.so` 试路由（不要与链入的 hook 同用）。`HS_BLAS_PROFILE=<文件>` 记录逐调用
  剖析，`HS_BLAS_NO_AVX512=1` / `HS_BLAS_NO_AVX2=1` 在 AVX-512 节点上模拟只有 AVX2 / 无 SIMD 的 CPU。命令与
  数据见 `docs/BLAS.md` §8。旧构建目录保留缓存的 `HSOLVE_BLAS_HOOK` / `HSOLVE_BLAS_ROUTE_DEFAULT`，须
  `-U` 后重新配置。

**所有编译之外的运行都必须经过 `tools/run/guard.py`**（下列默认值为开发服务器的；集群节点见上）：
- 默认绑定 NUMA node1 的 16 个物理核 `24-30,33-41`，进程树 RSS 上限 30 GiB；系统可用内存低于
  12 GiB 时只终止「大」进程树（≥ max(2 GiB, mem_gb/4)），小的构建/测试不受连带。RSS 上限（与外层
  guard 或调用方已声明的较小值取小）作为 `HSOLVE_MEMORY_LIMIT_GB` 导出给子进程，hsolve 的自动内存
  预算由它推出（§3.6-3）。
- 峰值内存报告取采样和、内核高水位 VmHWM 之和、ru_maxrss 三者最大值（不漏短时尖峰）。
- 重型实验加 `--exclusive`（全机串行）；单元测试用 `--skip-busy-check`。计时运行不要加
  `--skip-busy-check`：槽内的核若被别的进程占用，guard 会拒绝启动。
- 多 agent 并行阶段（开发服务器）：每个 agent 使用不相交的 `HS_CORES`（各 2 核）与 `HS_JOBS=2`，并设置
  `HS_MAX_SHARED_MEM_GB=8`（超过 8 GiB 的运行必须 `--exclusive`）。集群节点上改用上面的测量槽与构建道。
- 预估内存（MUMPS 分析）超过 28 GiB 的实验不运行，记录并上报。
- **MPI 绑核**：`mpiexec --bind-to core` 无视 guard 的 cpuset（40,41 下 SLEPc 的 rank 被绑到 CPU 0/48
  与 1/49）。需要绑核的 MPI 运行用 `tools/bench/baseline_slepc.py --bind core`（每个 rank 绑到继承
  cpuset 中自己的一块），或其它在继承 cpuset 内自绑的方式。
- 嵌套 guard（如 tune_slepc 每个试验一个 guard、外层再一个）：外层 `--exclusive` 持锁时向子进程导出
  `HS_SLOT_HELD`，内层 guard 共用这把锁（内层的 `--exclusive` 不再等锁，内存上限按单次运行上限算）。
  `tools/bench/compare.py` 即如此运行：`guard.py --exclusive --timeout ≤2700 -- python3 tools/bench/compare.py
  --cores 16-31 --solvers …`（compare.py 自己不加 `--exclusive`；`--solvers` 取 hsolve、hsolve-auto、
  hsolve-nocert、slepc，未知名称在任何运行之前被拒绝）。
  guard 收到 SIGTERM/SIGHUP 时终止整个进程树（被 nohup 忽略的 SIGHUP 仍然忽略）；外层超时终止时，
  内层的 mpiexec 进程组仍由内层 guard 负责。

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
| P3 | **已完成**：低内存输入装配与无映射铅笔；确定性正交化（融合 CGS2、DGKS、块 CGS2）；MUMPS 约定补充与 InertiaCounter；块 Krylov-Schur（eigen 层）；LOBPCG / JD；谱切片与惯性搜索的单边目标、零带切除、GHIEP 零簇移位放置；互斥分阶段计时；Python 接口；评测工具（汇总、合并参考、scipy 基线）。**未做**：SpMV SIMD；GHIEP 重特征值的块 Krylov（一般 Krylov-Schur 仍只支持 p = 1）——二者 P4 完成 |
| P4 | **已完成**（6a3d7c1）：稀疏核运行时分派（AVX-512）与融合修正残差、修正的舍入水平停止与活跃列回代、`FactorSolveOperator`；自研并行嵌套剖分（factor 层）与 ICNTL(15) 块分析；一般 Krylov-Schur 块模式与峰值列函数；src/verify（随机排除检验、单精度计数规则、代价模型）；driver：块 Krylov 默认（HEP/GHEP p = 4、GHIEP p = 2 的规则）、InertiaCounter 证书与 certificate 阶段、确定性排序模型（deterministic 用 METIS）、B 的 O(nnz) 筛查、GHIEP 后处理内存（wg-square-1e6 峰值 2.89 → 1.77 GB）；SLEPc 调优器与公平对比工具；集群节点部署。**未做**：JD + 单精度因子的内存受限回退；verify 接入 driver；自研嵌套剖分接入自适应排序；driver 检查单精度/BLR 修正是否收敛；16 核独占重测（SLEPc 调优与对比、排序、回代、GHIEP 块宽）；故障场景验证矩阵 |
| P5 | **已完成**（P5c，main 34cbfd4）：verify 接入 driver（certificate = exact / randomized / auto / off，单精度计数的双精度复核）；输入检查与 B 正定性检验链（分析前的廉价检验、IC + Gershgorin 严格证明）；自适应排序的 16 线程代价模型，自研嵌套剖分作为候选与 `ordering = 10`；σ₀ 处的零主元簇规则；块宽规则 `recommended_block_size` 与块步快路径 `try_fast`，nearest 运行按规则取块宽（内存只预留 P4 块宽的基）；收缩重跑以 `lead_nev` 在缺的对收敛后结束、自动块宽之后至多用 P4 块宽；certifier 的 InertiaCounter 推迟双精度实例的分析；GHIEP 内存预留按峰值列计、报告实际块宽；BLAS hook（src/blas，默认链入、route all）；以上都在 16 核测量槽下测量。**未做**：JD + 单精度因子的内存受限回退；SCOTCH 固定种子（之后可按其估计比较）；更好的嵌套剖分分隔子与 ND 功的预测（ss-tem152078）；一般 B 的 κ 证明（如分解 B − μ·diag(B)），使随机检验与单精度计数也适用于一般 HPD B；块宽规则在 BLAS hook 下的重新标定（现以 `kRuleWideMargin` 补偿，yee-bloch-1e4、yee-photonic-* 选 8 仍比 4 慢 2.5–3%）与用于切片；故障场景验证矩阵 |
| P7 | **已完成**（p7-membudget-fix）：内存预算默认 auto（cgroup、`HSOLVE_MEMORY_LIMIT_GB`、RLIMIT_AS/DATA、MemTotal 中最小的上限 L，0.85·L；地址空间上限按 VmSize 计并扣除已预留部分）；分析后按其常驻增量重新规划；内存失败改外存重试；ICNTL(23) 按 MUMPS 估计给出；单精度 / BLR 因子每次 apply 受检查、未达标以双精度同一主元顺序重解，`factor_precision = auto`（只在双精度放不进内存而单精度放得进时，可选）。32 GiB 下 1e6 三维测例改为内存内，快 1.71–1.83×（`results/p7_membudget.md`）。P6（混合精度求解，负结果）见分支 p6-mixprec-fix。 |
| P8 | **已完成**（p8-certcost-fix）：证书的计数失败不再结束求解——计数源（InertiaCounter / 主对象）失败时换另一个，计数前按计数余量检查估计（主对象代替计数器时按其上次分解的 INFOG(21)），SCOTCH 分析超上限时至多再分析两次；都不行时 nearest 交回已收敛的对、证书标为未完成（success = false），窗口目标只丢失败的切片；一般 HPD B 的 κ 与 λ_min 严格界（只用于 auto / randomized）；更便宜的精确计数全部实测为负（两进程并行、单次合并分解、省缩放/检测、换计数源），默认证书仍为 exact（`results/p8_certcost.md`）。 |
